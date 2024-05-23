import numpy as np
import torch
import pynvml
import torch.nn as nn
import json
from torch.nn.utils.stateless import functional_call
from collections import namedtuple
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import accuracy_score


Batch = namedtuple("Batch", ["x_sp", "y_sp", "x_qr", "y_qr"])


def set_device():
    pynvml.nvmlInit()
    num_gpus = pynvml.nvmlDeviceGetCount()
    gpu_free_memory = []

    if num_gpus > 0:
        for i in range(num_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = info.free  # Get free GPU memory in bytes
            gpu_free_memory.append((i, free_memory))

        pynvml.nvmlShutdown()
        gpu_free_memory.sort(key=lambda x: x[1], reverse=True)
        selected_gpu_indices = [gpu[0] for gpu in gpu_free_memory[:2]]

        # Set the selected GPU as the default device
        torch.cuda.set_device(selected_gpu_indices[0])
        device = torch.device('cuda')
        print(f'Device: {torch.cuda.current_device()}')
    else:
        print("No GPUs available. Using CPU.")
        device = torch.device('cpu')
        selected_gpu_indices = None
    return device, selected_gpu_indices


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def func_call(model, params_dict, args):
    if params_dict is None:
        params_dict = dict(model.named_parameters())
    y = functional_call(model, params_dict, args)
    return y


def save_object(object, name):
    if isinstance(object, dict):
        for k, v in object.items():
            obj = object[k]
            if isinstance(obj, np.ndarray):
                object[k] = v.tolist()
            if isinstance(obj, dict):
                for k2, v2 in object[k].items():
                    if isinstance(v2, np.ndarray):
                        object[k][k2] = v2.tolist()
    with open(name, 'w') as file:
        json.dump(object, file)

    return


def load_object(name):
    with open(name, 'r') as file:
        loaded_dict = json.load(file)
    return loaded_dict


def zeroed_gradients(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
    return


def define_task_labels(labels, num_classes):
    task_labels = []
    for i in range(0, len(labels), num_classes):
        sublist = labels[i:i+num_classes]
        task_labels.append(sublist)

    return task_labels


def count_parameters(model):
    nweights = []
    for name, param in model.named_parameters():
        if any(word in name.split('.') for word in model.pred_with_transformer):
            nweights.append(param.numel())
    return sum(nweights)


def increase_function(c0, current_step, max_steps=2000):
    if current_step >= max_steps:
        return 1.0
    else:
        return c0 + (current_step / max_steps) * (1.0 - c0)


class CustomLRScheduler(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, config_params, task_id):
        self.warmup_steps = config_params['warmup_steps']
        self.total_steps = config_params['steps']
        self.lr_init = config_params['lr']
        self.task_id = task_id
        super().__init__(optimizer)

    def get_lr(self):
        if self.task_id == 0:
            if self.last_epoch < self.warmup_steps:
                lr = self.lr_init * (self.last_epoch / self.warmup_steps)
            else:
                lr = self.lr_init * max(0.0, float(self.total_steps - self.last_epoch) / float(max(1, self.total_steps - self.warmup_steps)))
        else:
            lr = self.lr_init * max(0.0, float(self.total_steps - self.last_epoch) / float(max(1, self.total_steps)))
        return [lr for _ in self.optimizer.param_groups]


def set_optimizer(model, transformer_model, config_params):
    parameters_model = {n: w for n, w in model.named_parameters() if not any(word in n.split('.') for word in model.pred_with_transformer)}
    parameters_transformer = transformer_model.parameters()

    model_optimizer = torch.optim.Adam(parameters_model.values(), lr=config_params['lr'], weight_decay=config_params['lambda_l2'])
    transformer_optimizer = torch.optim.Adam(parameters_transformer, lr=config_params['lr'], weight_decay=config_params['lambda_l2'])

    return model_optimizer, transformer_optimizer


def l2_regularization(dict_parameters):
    l2reg_loss = 0.0
    for param in dict_parameters.values():
        l2reg_loss += torch.norm(param, p=2)
    return l2reg_loss


def compute_importance_score(model, theta, batch, criterion, config_params):
    model_params = dict(model.named_parameters())
    model_params.update(theta['model'])

    zeroed_gradients(model)
    pred = func_call(model, model_params, batch.x_sp)
    loss = criterion(pred, batch.y_sp)
    loss.backward()

    weights_info = estimate_importance(model)

    if config_params['top_K'] != 1.0:
        weights_info['score'] = select_top_k(weights_info['score'], config_params['top_K'])
    return weights_info


def estimate_importance(model):
    weights_info = {'data': {}, 'grad': {}, 'score': {}}
    min_importance = float('inf')
    max_importance = float('-inf')
    for name, param in model.named_parameters():
        if not any(word in name.split('.') for word in model.pred_with_transformer):
            param.grad = None
            continue
        elif param.requires_grad and param.grad is not None:
            weights_info['data'][name] = param.data.clone()
            weights_info['grad'][name] = param.grad.clone()
            importance = abs(weights_info['data'][name]*weights_info['grad'][name])
            weights_info['score'][name] = importance
            min_importance = min(min_importance, torch.min(importance))
            max_importance = max(max_importance, torch.max(importance))
            param.grad = None

    # Normalize in [0, 1]
    for name, importance in weights_info['score'].items():
        normalized_importance = (importance - min_importance) / (max_importance - min_importance)
        weights_info['score'][name] = normalized_importance

    return weights_info


def select_top_k(importance_scores, K):
    new_importance_scores = importance_scores.copy()
    for name, scores in importance_scores.items():
        flattened_scores = torch.cat([i.view(-1) for i in scores])
        sorted_scores, _ = torch.sort(flattened_scores, descending=True)

        threshold_index = int(len(sorted_scores) * K)
        threshold_value = sorted_scores[threshold_index]

        mask = flattened_scores >= threshold_value
        mask = mask.view(scores.shape)

        new_importance_scores[name] = scores * mask
    return new_importance_scores


def update_model_parameters(model, weight_updates):
    # Counter for keeping track of where we are in the updates tensor
    update_counter = 0
    updated_parameters = {}
    for name, param in model.named_parameters():
        if any(word in name.split('.') for word in model.pred_with_transformer):
            param_data = param.data
            param_shape = param.shape
            param_size = param.numel()
            layer_update = weight_updates[update_counter: update_counter + param_size].reshape(param_shape)
            param_data += layer_update
            updated_parameters[name] = param_data.clone()

            update_counter += param_size
    assert update_counter == len(weight_updates), "Different number of weights and weight updates"
    return updated_parameters


def accuracy(pred, y_true):
    y_pred = pred.argmax(1).reshape(-1).cpu()
    y_true = y_true.reshape(-1).cpu()
    return accuracy_score(y_pred, y_true)


def compute_bwt(tasks_accuracies):
    forget_rates = {}
    all_bwt = 0.0
    n = 0.0
    for T in tasks_accuracies.keys():
        forget_rates[T] = {}
        for i in range(T):
            if i in tasks_accuracies[T] and i in tasks_accuracies[i]:
                acc_current_time = tasks_accuracies[T][i]
                acc_original_time = tasks_accuracies[i][i]
                forget = acc_current_time-acc_original_time
                forget_rates[T][i] = forget
                all_bwt += forget
                n += 1
    average_forget_rate = all_bwt / n
    return average_forget_rate


def compute_fwt(tasks_accuracies):
    n_tasks = len(tasks_accuracies)
    forget_rates = {}
    all_fwt = 0.0
    n = 0.0
    for T in tasks_accuracies.keys():
        for i in range(T, n_tasks):
            if i in tasks_accuracies[T] and i in tasks_accuracies[i]:
                future_acc = tasks_accuracies[T][i]
                all_fwt += future_acc
                n += 1
    average_fwt = all_fwt / n
    return average_fwt


