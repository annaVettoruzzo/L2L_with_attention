import numpy as np
import random
import copy
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import functional as F
from utils import Batch
from cifar_dataset import CIFAR100


class RotateTransform:
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return F.rotate(x, self.angle)


def get_train_loader(train_dataset, batch_size):
    train_loader = DataLoader(train_dataset,
                              batch_size,
                              num_workers=0,
                              pin_memory=True,
                              shuffle=True)
    return train_loader


def get_test_loader(test_dataset, test_batch_size):
    test_loader = DataLoader(test_dataset,
                             test_batch_size,
                             shuffle=False,
                             num_workers=0,
                             pin_memory=True)
    return test_loader


def load_data(dataset):
    if 'mnist' in dataset:
        if dataset == 'permutedmnist':
            full_dataset, test_dataset = torch.load('data/PMNIST/mnist_permutations.pt')
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                            ])
            full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    elif dataset == "cifar100":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                             (0.2675, 0.2565, 0.2761))
                                        ])

        full_dataset = CIFAR100('./data/CIFAR100', train=True, transform=transform, download=False)
        test_dataset = CIFAR100('./data/CIFAR100', train=False, transform=transform, download=False)
    raise ValueError("Code implemented only for 'mnist' or 'cifar'")
    return full_dataset, test_dataset


def split_task_construction(dataset, task_labels, class_il=False):
    full_dataset, test_dataset = load_data(dataset)
    train_datasets, test_datasets = [], []
    task_idx = 0
    for labels in task_labels:
        # Map each class to a label in {0, ..., n-1} consistently in train and test
        new_labels = random.sample(range(len(task_labels[task_idx])), len(task_labels[task_idx]))
        label_map = dict(zip(labels, new_labels))
        if class_il:  # for class-il we do not want to do so
            label_map = dict(zip(labels, labels))
        splitted_train_dataset = create_split_dataset(full_dataset, labels, label_map)
        splitted_test_dataset = create_split_dataset(test_dataset, labels, label_map)
        train_datasets.append(splitted_train_dataset)
        test_datasets.append(splitted_test_dataset)
        task_idx += 1
    return train_datasets, test_datasets


def create_split_dataset(dataset, labels, label_map):
    idx = np.in1d(dataset.targets, labels)
    splitted_dataset = copy.deepcopy(dataset)
    if isinstance(splitted_dataset.targets, list):
        splitted_dataset.targets = torch.FloatTensor(splitted_dataset.targets)
    splitted_dataset.targets = splitted_dataset.targets[idx]
    splitted_dataset.targets = torch.from_numpy(np.array([label_map[int(label)] for label in splitted_dataset.targets]))
    if isinstance(splitted_dataset, datasets.ImageFolder):
        new_imgs = []
        for i, include in enumerate(idx):
            if include:
                img_path, old_label = splitted_dataset.imgs[i]
                new_imgs.append((img_path, label_map[int(old_label)]))
        splitted_dataset.imgs = new_imgs
        splitted_dataset.samples = new_imgs
    else:
        splitted_dataset.data = splitted_dataset.data[idx]
    return splitted_dataset


def rotated_task_construction(n_tasks):
    train_datasets, test_datasets = [], []
    rotation_angles = []
    mmin_rot, mmax_rot = 0., 180.
    for t in range(n_tasks):
        min_rot = 1.0 * t / n_tasks * (mmax_rot - mmin_rot) + mmin_rot
        max_rot = 1.0 * (t + 1) / n_tasks * (mmax_rot - mmin_rot) + mmin_rot
        rot = round(random.random() * (max_rot - min_rot) + min_rot)

        transform = transforms.Compose([RotateTransform(rot),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
        train_datasets.append(datasets.MNIST('./data', train=True, download=True, transform=transform))
        test_datasets.append(datasets.MNIST('./data', train=False, download=True, transform=transform))
        rotation_angles.append(rot)
    return list(train_datasets), list(test_datasets), list(rotation_angles)


def permuted_task_construction(dataset):
    train_dataset, test_dataset = load_data(dataset)
    n_tasks = len(train_dataset)
    task_permutation = torch.randperm(n_tasks).tolist()
    permuted_train_dataset, permuted_test_dataset = [], []
    for idx in task_permutation:
        train_tsk = train_dataset[idx][1:]
        test_tsk = test_dataset[idx][1:]
        permuted_train_dataset.append(train_tsk)
        permuted_test_dataset.append(test_tsk)
    return permuted_train_dataset, permuted_test_dataset


class BatchGenerator:
    def __init__(self, task_dataset, config_params):
        self.data = task_dataset
        self.config_params = config_params
        self.num_classes = config_params['n']
        self.k = config_params['k']
        self.q = 100

        self.ds_dict = defaultdict(list)
        if config_params['dataset'] == 'permutedmnist':
            for img, c in zip(self.data[0], self.data[1]):
                self.ds_dict[int(c)].append(img)
        else:
            for img, c in task_dataset:
                self.ds_dict[int(c)].append(img)
        for c in self.ds_dict.keys():
            self.q = min(self.q, len(self.ds_dict[c]) - self.k)

    def get_batch(self, device=None):
        if device is None:
            device = self.config_params['device']

        classes = list(self.ds_dict.keys())
        label_map = dict(zip(classes, classes))
        if self.config_params['method'] == 'maml' or self.config_params['method'] == 'proposed':
            # Randomly map each selected class to a random label in {0, ..., n-1}
            labels = random.sample(range(self.num_classes), self.num_classes)
            label_map = dict(zip(classes, labels))

        # Randomly select k support examples and q query examples from each of the selected classes
        x_sp, y_sp, x_qr, y_qr = [], [], [], []
        for c in classes:
            images = random.sample(self.ds_dict[c], self.k + self.q)
            x_sp += images[:self.k]
            y_sp += [label_map[c] for _ in range(self.k)]
            x_qr += images[self.k:]
            y_qr += [label_map[c] for _ in range(self.q)]

        # Transform these lists to appropriate tensors and return them
        x_sp, y_sp, x_qr, y_qr = [torch.from_numpy(np.array(lst)).to(device).float()
                                  for lst in [x_sp, y_sp, x_qr, y_qr]]
        y_sp, y_qr = y_sp.long(), y_qr.long()

        return Batch(x_sp, y_sp, x_qr, y_qr)
