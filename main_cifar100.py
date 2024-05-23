import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import pickle
import dataset
from models import base_models, transformer_models, base_models_wsn, base_models_protonet
import utils
from methods import PROPOSED, MAML, ProtoNet, EWC, SI, LAMAML, SparseMAML, WSN, ICARL, DERPP
import test
import make_plots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def main(args):
    # Set default parameters
    config_params = vars(args)
    print(f"Method: {config_params['method']}")

    # For saving models
    if config_params['multihead']:
        PATH = Path(f"_saved_models_final/{config_params['dataset']}/seed{config_params['seed']}/{config_params['method']}")
    else:
        PATH = Path(f"_saved_models_final/{config_params['dataset']}/seed{config_params['seed']}/{config_params['method']}")
    PATH.mkdir(parents=True, exist_ok=True)

    torch.random.manual_seed(config_params['seed'])
    np.random.seed(config_params['seed'])
    random.seed(config_params['seed'])

    device, selected_gpus = utils.set_device()
    config_params['device'] = device

    # Set dataset
    print("Loading the dataset...")
    all_labels = list(range(config_params['num_labels']))
    task_labels = utils.define_task_labels(all_labels, config_params['n'])
    task_labels = task_labels[:5]  # increase this to set the number of tasks in the stream
    n_tasks = len(task_labels)
    train_dataset, test_dataset = dataset.split_task_construction(config_params['dataset'], task_labels)

    # Model
    print("Initializing the models...")
    criterion = nn.CrossEntropyLoss().to(config_params['device'])

    # Benchmarks
    n_heads = n_tasks if config_params['multihead'] else None
    if config_params['method'] == 'proposed':
        n_heads = None  # always single head
        model = base_models.ThreeConvNetSimple(config_params, n_heads, n_classes=config_params['n']).to(config_params['device'])
        nweights = utils.count_parameters(model)
        print(nweights)
        transformer_model = transformer_models.TransformerModel(nweights, config_params).to(config_params['device'])
        method = PROPOSED(model, transformer_model, criterion, config_params)
    else:
        raise ValueError('Invalid method: {}'.format(config_params['method']))

    # Training
    previous_tasks_acc, future_tasks_acc = {}, {}
    print("Start training...")
    for task_idx in range(n_tasks):
        print(f"Task idx: {task_idx}")
        save_dir = Path(PATH / f'task{int(task_idx)}')
        save_dir.mkdir(parents=True, exist_ok=True)

        method, weights_info = method.fit(train_dataset[task_idx], task_idx, save_dir=save_dir)

        if config_params['method'] == 'proposed':
            # Save weights info
            with open(save_dir / 'task_scores.pkl', 'wb') as f:
                pickle.dump(weights_info['score'], f)
            with open(save_dir / 'task_weights.pkl', 'wb') as f:
                pickle.dump(weights_info['data'], f)
            with open(save_dir / 'task_grads.pkl', 'wb') as f:
                pickle.dump(weights_info['grad'], f)
            with open(save_dir / 'task_updates.pkl', 'wb') as f:
                pickle.dump(weights_info['update'], f)

        # Test the model on previous tasks
        #device_val = f'cuda:{selected_gpus[1]}' if len(selected_gpus) > 1 else device
        device_val=device
        previous_tasks_acc[int(task_idx)] = test.test_previous_tasks(method, test_dataset, task_idx, device_val)
        utils.save_object(previous_tasks_acc, save_dir / 'test_accuracy.json')

        if task_idx+1 < n_tasks:
            # Test the model on future tasks
            future_tasks_acc[int(task_idx)] = test.test_future_tasks(method, test_dataset, task_idx, n_tasks, device_val)

    bwt = utils.compute_bwt(previous_tasks_acc)
    print('BWT after {:} steps: {:.4f}'.format(config_params['test_steps'], bwt))
    previous_tasks_acc["bwt"] = round(bwt, 4)

    fwt = utils.compute_fwt(future_tasks_acc)  
    print('FWT after {:} steps: {:.4f}'.format(config_params['test_steps'], fwt))
    previous_tasks_acc["fwt"] = round(fwt, 4)

    torch.save(method.model.state_dict(), PATH / 'model.pth')
    if config_params['method'] == 'proposed':
        torch.save(method.transformer_model.state_dict(), PATH / 'transformer_model.pth')
    utils.save_object(previous_tasks_acc, PATH / 'test_accuracy.json')
    make_plots.make_accuracy_matrix(previous_tasks_acc, n_tasks, PATH)

if __name__ == '__main__':
    # Set parameters
    parser = argparse.ArgumentParser(description='Meta-Learning for Continual Learning')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    # Set dataset params
    parser.add_argument('--method', type=str, default='proposed',
                        help='Benchmark name')
    parser.add_argument("--multihead", action="store_true",
                        help="Enable multihead")
    parser.add_argument('--dataset', type=str, default='cifar100',
                        help='Dataset name')
    parser.add_argument('--input_dim', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--input_size', type=int, default=32,
                        help='Size of images')
    parser.add_argument('--num_labels', type=int, default=100,
                        help='Total number of labels')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--k', type=int, default=5,
                        help='Number of images per class in support set')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of classes per task')
    # Set training params
    parser.add_argument('--steps', type=int, default=100000,
                        help='Number of steps per task')
    parser.add_argument('--inner_steps', type=int, default=5,
                        help='Number of inner steps')
    parser.add_argument('--test_steps', type=int, default=50,
                        help='Number of test steps for adaptation')
    parser.add_argument('--warmup_steps', type=int, default=3000,
                        help='Number of warmup steps for the first task')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr_inner', type=float, default=1e-3,
                        help='Learning rate inner loop')
    parser.add_argument('--lr_inner_test', type=float, default=1e-5,
                        help='Learning rate inner loop at test time')
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='Momentum for SGD')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout (transformer)')
    parser.add_argument('--lambda_l2', type=float, default=1e-5,
                        help='Lambda value for L2 regularization')
    parser.add_argument('--c', type=float, default=0.8,
                        help='Parameter to retain previous task knowledge')
    # Set model parameters
    parser.add_argument('--features_dim', type=int, default=100,
                        help='Number of CNN filters')
    parser.add_argument('--task_encoder', type=bool, default=True,
                        help='Use task encoder')
    parser.add_argument('--embedding_size', type=int, default=16,
                        help='Embedding size')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of heads (transformer)')
    parser.add_argument('--dim_feedforward', type=int, default=64,
                        help='Dimension of the feedforward network model (transformer)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of encoder layers (transformer)')
    parser.add_argument('--use_bn', type=bool, default=True,
                        help='Use batch normalization in base learner')
    parser.add_argument('--predict_bn', type=bool, default=False,
                        help='Predict batch normalization parameters')
    parser.add_argument('--unique_te', type=bool, default=True,
                        help='Use a unique vector as task embedding')
    parser.add_argument('--top_K', type=float, default=0.4,
                        help='Keep only top-K importance scores (K is the percentage)')
    parser.add_argument('--clamping_transformer', type=float, default=3.0,
                        help='Clamping parameters for transformer output')
    parser.add_argument('--spatial', type=int, default=10000,
                        help='Set to the number of spatial connections')

    args = parser.parse_args()

    main(args)