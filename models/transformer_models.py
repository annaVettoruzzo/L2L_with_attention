import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models import task_encoder_models


class TransformerModel(nn.Module):
    def __init__(self, nweights, config_params, pretrained_model_path=None):
        super().__init__()
        self.d_model = config_params['embedding_size']
        self.nhead = config_params['num_heads']
        self.d_hid = config_params['dim_feedforward']
        self.nlayers = config_params['n_layers']
        self.dropout = config_params['dropout']
        self.nweights = nweights
        self.spatial = config_params['spatial']
        self.clamp = config_params['clamping_transformer']
        self.device = config_params['device']

        te_input_dim = 1
        if config_params['dataset'] == 'permutedmnist':
            self.te_model = task_encoder_models.TaskEncoderPMNIST(config_params)
        elif 'mnist' in config_params['dataset']:
            self.te_model = task_encoder_models.TaskEncoderMNIST(pretrained_model_path, config_params).to(self.device)
        elif 'cifar' in config_params['dataset'] or 'imagenet' in config_params['dataset']:
            self.te_model = task_encoder_models.TaskEncoderCIFAR(config_params).to(self.device)
        encoder_layers = TransformerEncoderLayer(self.d_model, self.nhead, self.d_hid, self.dropout, activation="gelu").to(self.device)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers).to(self.device)
        self.embedding = FeatureExtractor(self.d_model-1).to(self.device)  # create feature embedding

        if self.spatial is None:
            self.pos_encoder = PositionalEncoding(te_input_dim, self.nweights, self.d_model).to(self.device)
            self.linear = nn.Linear(self.d_model, 9).to(self.device)
        else:
            self.pos_encoder = PositionalEncoding(te_input_dim, self.spatial, self.d_model).to(self.device)
            self.linear_task = nn.Linear(self.d_model, 1).to(self.device)
            self.linear_weights = nn.Linear(self.d_model, 9).to(self.device)

        self.tanh = nn.Tanh()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.spatial is None:
            self.linear.bias.data.zero_()
            self.linear.weight.data.uniform_(-initrange, initrange)
        else:
            self.linear_task.bias.data.zero_()
            self.linear_task.weight.data.uniform_(-initrange, initrange)
            self.linear_weights.bias.data.zero_()
            self.linear_weights.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, weights, importance_scores, src_mask=None, device=None):
        """
        Forward function where weights (and score) are input per filter (spatial allocation) if spatial is not None --> we consider the structure of conv2d.
        In this architecture weights are input in the feature extractor and then a single score is concatenated to each input entry in the sequence.
        The "single score" is computed by averaging the scores of all weights in the same filter.

        Arguments:
            x: support set
            weights: weights of the base model
            importance_scores: vector with an importance score per each weight
            src_mask: mask for the source sequence ensures that position i is allowed to attend tha un,asked positions.
            device: gpu device 
        """
        task_encoding = self.te_model(x)
        task_encoding = task_encoding.unsqueeze(0) if task_encoding.dim() == 1 else task_encoding

        if self.spatial is None:
            all_model_weights = torch.cat([w.view(-1) for w in weights.values()]).unsqueeze(1)  # shape [seq_len, batch_size]
            all_importance_scores = torch.cat([i.view(-1) for i in importance_scores.values()]).unsqueeze(1)  # shape [seq_len, batch_size
        else:
            all_model_weights, all_importance_scores = [], []
            for n, w in weights.items():
                if 'weight' in n:
                    out_ch, in_ch, k, _ = w.shape
                    new_w = w.view(out_ch * in_ch, k * k)
                    new_s = importance_scores[n].view(out_ch * in_ch, k * k)

                    all_model_weights.append(new_w)
                    all_importance_scores.append(new_s)

            all_model_weights = torch.cat(all_model_weights)  # shape [seq_length, filter_size^2]
            all_importance_scores = torch.stack([torch.mean(score) for score in all_importance_scores[0]])
            all_importance_scores = all_importance_scores.unsqueeze(1)

        all_model_weights = self.embedding(all_model_weights)  # shape [seq_length, d_model-1]
        all_model_weights = torch.cat([all_model_weights, all_importance_scores], dim=1)  # shape [seq_length, d_model-1+1]
        src = torch.cat([task_encoding, all_model_weights])  # src: shape ``[nweights+1, d_model]``
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)

        if self.spatial is None:
            output = self.linear(output)
            output_task, output_weights = output[:task_encoding.shape[0]], output[task_encoding.shape[0]:]
        else:
            output_task, output_weights = output[:task_encoding.shape[0]], output[task_encoding.shape[0]:]
            output_task = self.linear_task(output_task)
            output_weights = self.linear_weights(output_weights).view(-1)

        output_weights = self.tanh(output_weights) * self.clamp

        torch.cuda.empty_cache()
        return output_task, output_weights


class PositionalEncoding(nn.Module):
    def __init__(self, te_input_dim, size, d_model):
        super().__init__()

        self.pos_embedding = nn.Parameter(torch.empty(te_input_dim + size, d_model).normal_(std=0.2))

    def forward(self, x):
        x = x + self.pos_embedding
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(nn.LazyLinear(d_model),
                                 nn.ReLU(),
                                 nn.Linear(d_model, d_model))

    def forward(self, x):
        return self.net(x)


