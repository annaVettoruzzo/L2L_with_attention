import torch
import copy
import numpy as np
import dataset
import utils


def estimate_importance(c, model):
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
            if isinstance(c, dict):
                importance = weights_info['grad'][name] * c[name]
            else:
                importance = weights_info['grad'][name]
            weights_info['score'][name] = importance
            min_importance = min(min_importance, torch.min(importance))
            max_importance = max(max_importance, torch.max(importance))
            param.grad = None

    # Normalize in [0, 1]
    for name, importance in weights_info['score'].items():
        normalized_importance = (importance - min_importance) / (max_importance - min_importance)
        weights_info['score'][name] = normalized_importance
    return weights_info


class PROPOSED:
    def __init__(self, model, transformer, criterion, config_params):
        self.model = model
        self.transformer_model = transformer
        self.criterion = criterion
        self.config_params = config_params

        self.model_optimizer, self.transformer_optimizer = utils.set_optimizer(self.model, self.transformer_model, config_params)

        self.theta = {
            'model': {n: w for n, w in self.model.named_parameters() if not any(word in n.split('.') for word in self.model.pred_with_transformer)},
            'transformer': {n: w for n, w in self.transformer_model.named_parameters() if w.requires_grad}
        }

        self.previous_tsk_score = None
        self.task = 0

    def compute_c(self, step, tot_steps):
        c = dict()
        for name, param in self.model.named_parameters():
            if any(word in name.split('.') for word in self.model.pred_with_transformer):
                if self.task <= 1 or step > 2*tot_steps/3:
                    c[name] = torch.ones_like(param)
                    c_value = 1.0
                else:
                    c[name] = torch.where(self.previous_tsk_score[name] < 1e-20, 1.0, self.config_params['c'] + ((1-self.config_params['c']) * step / (2*tot_steps/3)))
                    c_value = self.config_params['c'] + ((1-self.config_params['c']) * step / (2*tot_steps/3))
        return c, c_value

    def compute_importance_score(self, theta, batch, c, model=None):
        model_params = dict(model.named_parameters())
        model_params.update(theta['model'])

        utils.zeroed_gradients(model)
        pred = utils.func_call(model, model_params, batch.x_sp)
        loss = self.criterion(pred, batch.y_sp)
        loss.backward()

        weights_info = estimate_importance(c, model)

        if self.config_params['top_K'] != 1.0 and self.task > 0:  # if it is the first task we predict all weights
            weights_info['score'] = utils.select_top_k(weights_info['score'], self.config_params['top_K'])
        return weights_info

    def update_model_parameters(self, weight_updates, model):
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

    def get_updated_params(self, params_dict, batch, c, model=None, transformer_model=None, device=None):
        if model is None:
            model = self.model
        if transformer_model is None:
            transformer_model = self.transformer_model

        # Compute importance scores
        weights_info = self.compute_importance_score(params_dict, batch, c, model)
        # Predict parameters' update
        transformer_params = dict(transformer_model.named_parameters())
        transformer_params.update(params_dict['transformer'])
        output_task, weight_updates = utils.func_call(transformer_model, transformer_params,
                                                      (batch.x_sp, weights_info['data'], weights_info['score'], None, device))
        del transformer_params
        weights_info['update'] = weight_updates
        # Update model's parameters
        updated_model_params = self.update_model_parameters(weight_updates, model)
        updated_model_params.update(params_dict['model'])
        return output_task, updated_model_params, weights_info

    def one_inner_loop(self, params_dict, batch, task_id, c, device=None):
        output_task, updated_model_params, weights_info = self.get_updated_params(params_dict, batch, c, self.model, self.transformer_model, device)
        # Make predictions
        predictions = utils.func_call(self.model, updated_model_params, batch.x_sp)
        # Compute inner loss
        inner_loss = self.criterion(predictions, batch.y_sp)
        dummy_loss = torch.nn.MSELoss()(output_task.squeeze(), task_id)  # loss for the transformer_model to attach gradients
        l2reg = utils.l2_regularization(updated_model_params)
        inner_loss = inner_loss + 1e-3*dummy_loss + self.config_params['lambda_l2'] * l2reg

        # Compute new theta parameters after GD step
        all_params_dict = {**params_dict['model'], **params_dict['transformer']}
        grads = torch.autograd.grad(inner_loss, all_params_dict.values())

        new_params_dict = {'model': {}, 'transformer': {}}
        for (name, w), w_grad in zip(all_params_dict.items(), grads):
            if name in updated_model_params.keys():
                if not any(word in name.split('.') for word in self.model.pred_with_transformer):
                    new_params_dict['model'][name] = w - self.config_params['lr_inner'] * w_grad
            else:
                new_params_dict['transformer'][name] = w - self.config_params['lr_inner'] * w_grad
        del updated_model_params
        return new_params_dict, weights_info, inner_loss

    def inner_loop(self, batch, task_id, c, device=None):
        new_theta, _, _ = self.one_inner_loop(self.theta, batch, task_id, c, device=device)
        for _ in range(self.config_params['inner_steps'] - 2):
            new_theta, _, _ = self.one_inner_loop(new_theta, batch, task_id, c, device=device)
        return new_theta

    def fit(self, data, task_id, save_dir=None, print_output=True):
        self.task = task_id
        task_id = torch.tensor(task_id, dtype=torch.float).to(self.config_params['device'])

        transformer_scheduler = utils.CustomLRScheduler(self.transformer_optimizer, self.config_params, self.task)
        current_task_steps = self.config_params['warmup_steps'] + self.config_params['steps'] if task_id == 0 else self.config_params['steps']

        batch_generator = dataset.BatchGenerator(data, self.config_params)

        for step in range(current_task_steps):
            # encourage the model to select different weights from the previous ones at the beginning of the training
            c, c_value = self.compute_c(step, tot_steps=current_task_steps)

            batch = batch_generator.get_batch()
            new_theta = self.inner_loop(batch, task_id, c)
            output_task, updated_model_params, weights_info = self.get_updated_params(new_theta, batch, c)
            updated_model_params.update(new_theta['model'])
            predictions = utils.func_call(self.model, updated_model_params, batch.x_qr)

            # Compute query loss
            inner_loss = self.criterion(predictions, batch.y_qr)
            dummy_loss = torch.nn.MSELoss()(output_task.squeeze(), task_id)
            l2reg = utils.l2_regularization(updated_model_params)
            loss = inner_loss + 1e-3*dummy_loss + self.config_params['lambda_l2']*l2reg

            self.model_optimizer.zero_grad()
            self.transformer_optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(self.transformer_model.parameters(), 1.0)

            self.model_optimizer.step()
            self.transformer_optimizer.step()

            if print_output:
                if (step+1) % 200 == 0:
                    current_lr = self.transformer_optimizer.param_groups[0]['lr']  # Assuming lr is in the first param_group
                    print('Train Step: {}. Loss: {:.4f}, Inner_loss: {:.4f}, Dummy loss: {:.4f}, l2reg: {:.4f}, lr: {:.6f}, c: {:.1f}'.format(
                        step+1, loss.item(), inner_loss.item(), dummy_loss.item(), l2reg.item(), current_lr, c_value))

            if (step+1) % 5000 == 0 or (step+1) % current_task_steps == 0:
                if save_dir is not None:
                    torch.save({
                        'step': step+1,
                        'model_state_dict': self.model.state_dict(),
                        'transformer_state_dict': self.transformer_model.state_dict(),
                        'model_optimizer_state_dict': self.model_optimizer.state_dict(),
                        'transformer_optimizer_state_dict': self.transformer_optimizer.state_dict(),
                    }, save_dir/f'model_checkpoint_step_{step+1}.pth')

            transformer_scheduler.step()

        self.previous_tsk_score = weights_info['score']
        return self, weights_info

    def adapt_and_evaluate(self, batch, task_id, device=None, print_output=False):
        if device is None:
            device = self.config_params['device']
        task_idx = torch.tensor(task_id, dtype=torch.float).to(device)
        cmodel = copy.deepcopy(self.model).to(device)
        ctransformer_model = copy.deepcopy(self.transformer_model).to(device)

        theta = {
            'model': {n: w for n, w in cmodel.named_parameters() if not any(word in n.split('.') for word in cmodel.pred_with_transformer)},
            'transformer': {n: w for n, w in ctransformer_model.named_parameters() if w.requires_grad}
        }
        optimizer = torch.optim.SGD([{'params': theta['model'].values()},
                                     {'params': theta['transformer'].values()}],
                                    lr=config_params['lr_inner'], momentum=self.config_params['momentum'])
        test_loss = []
        test_accuracy = []
        for step in range(self.config_params['test_steps']):
            optimizer.zero_grad()
            predictions = utils.func_call(cmodel, None, batch.x_qr)
            loss = self.criterion(predictions, batch.y_qr)
            acc = utils.accuracy(predictions, batch.y_qr)
            test_accuracy.append(acc)
            test_loss.append(loss.item())

            # Adapt the models using training data
            output_task, updated_model_params, weights_info = self.get_updated_params(theta, batch, c=None, model=cmodel, transformer_model=ctransformer_model, device=device)
            predictions = utils.func_call(cmodel, updated_model_params, batch.x_sp)
            inner_loss = self.criterion(predictions, batch.y_sp)
            dummy_loss = torch.nn.MSELoss()(output_task.squeeze(), task_idx)  # loss for the transformer_model to attach gradients
            l2reg = utils.l2_regularization(updated_model_params)
            loss = inner_loss + 1e-3 * dummy_loss + self.config_params['lambda_l2'] * l2reg

            loss.backward()
            optimizer.step()

        if print_output:
            print('\nEvaluation: Final loss: {:.4f}, Final accuracy: {:.3f}%\n'.format(
               test_loss[-1], 100. * test_accuracy[-1]))

        return test_accuracy, test_loss 

    def evaluate(self, data, task_id, nb_test_batch=5, device=None):
        test_generator = dataset.BatchGenerator(data, self.config_params)
        avg_accuracy = []
        for _ in range(nb_test_batch):
            batch = test_generator.get_batch(device)
            test_accuracy, test_loss = self.adapt_and_evaluate(batch, task_id, device)
            avg_accuracy.append(test_accuracy)
        avg_accuracy = np.array(avg_accuracy).mean(axis=0)
        return avg_accuracy[-1]





