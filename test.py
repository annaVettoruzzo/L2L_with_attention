import numpy as np
import dataset
from methods import WSN, PROPOSED


def test_previous_tasks(method, test_dataset, task_idx, device=None):
    """ Backward transfer """
    previous_task_acc = {}
    for tsk_id in range(task_idx+1):
        if isinstance(method, WSN):
            _, test_accuracy = method.evaluate(test_dataset[tsk_id], tsk_id, method.per_task_masks[tsk_id], mode='test')
        else:
            test_accuracy = method.evaluate(test_dataset[tsk_id], tsk_id, device=device)
        previous_task_acc[int(tsk_id)] = np.around(test_accuracy, 4)
    avg_task_acc = sum(previous_task_acc.values()) / len(previous_task_acc)
    print('Average accuracy over {} tasks: {:}%'.format(task_idx + 1, 100. * avg_task_acc))
    return previous_task_acc


def test_future_tasks(method, test_dataset, task_idx, n_tasks, device=None):
    """ Forward transfer """
    future_task_acc = {}
    for tsk_id in range(task_idx, n_tasks):
        if isinstance(method, WSN):
            _, test_accuracy = method.evaluate(test_dataset[tsk_id], tsk_id, curr_task_masks=None, mode='train')  # here we use mode='train' to create a new mask for future tasks
        else:
            test_accuracy = method.evaluate(test_dataset[tsk_id], tsk_id, device=device)
        future_task_acc[int(tsk_id)] = np.around(test_accuracy, 4)
    avg_task_acc = sum(future_task_acc.values()) / len(future_task_acc)
    return future_task_acc