import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self, config_params, n_heads=None, n_classes=None):
        super().__init__()
        self.input_dim = config_params['input_dim']
        self.features_dim = config_params['features_dim']
        self.num_classes = n_classes
        self.config_params = config_params
        self.n_heads = n_heads

        self.conv1 = nn.Conv2d(self.input_dim, self.features_dim, 3, padding='same', bias=False)
        if self.config_params['use_bn']:
            self.bn1 = nn.BatchNorm2d(self.features_dim, track_running_stats=False)
        self.act = nn.LeakyReLU() 
        self.maxpool = nn.MaxPool2d(2)
        self.flat = torch.nn.Flatten()

        # account for multi-head approach
        dim_flattened = int(self.features_dim * int(self.config_params['input_size']/2) * int(self.config_params['input_size']/2))
        self.last = nn.Linear(dim_flattened, self.num_classes, bias=False) if n_heads is None \
            else nn.ModuleList([nn.Linear(dim_flattened, self.num_classes, bias=False) for _ in range(n_heads)])

        self.initialize_weights()

        self.pred_with_transformer = ['conv1']
        if config_params['predict_bn']:
            self.pred_with_transformer.append('bn1')

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # Apply Xavier initialization to weights of convolutional and linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                # Initialize batch normalization layers with a small variance to avoid scale issues
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x, task_idx=None):
        x = self.conv1(x)
        if self.config_params['use_bn']:
            x = self.bn1(x)
        x = self.maxpool(self.act(x))
        x = self.flat(x)
        # account for multi-head approach (for TIL baselines)
        out = self.last(x) if self.n_heads is None else self.last[task_idx](x)
        return out


class ThreeConvNetSimple(nn.Module):
    def __init__(self, config_params, n_heads=None, n_classes=None):
        super().__init__()
        self.input_dim = config_params['input_dim']
        self.features_dim = config_params['features_dim']
        self.num_classes = n_classes
        self.config_params = config_params
        self.n_heads = n_heads

        self.conv1 = nn.Conv2d(self.input_dim, self.features_dim, 3, bias=False)
        self.conv2 = nn.Conv2d(self.features_dim, self.features_dim, 3, bias=False)
        self.conv3 = nn.Conv2d(self.features_dim, self.features_dim, 3, bias=False)
        if self.config_params['use_bn']:
            self.bn1 = nn.BatchNorm2d(self.features_dim, track_running_stats=False)
            self.bn2 = nn.BatchNorm2d(self.features_dim, track_running_stats=False)
            self.bn3 = nn.BatchNorm2d(self.features_dim, track_running_stats=False)

        self.act = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.flat = torch.nn.Flatten()
        dim_flatten = self.features_dim * int(self.config_params['input_size']/16) * int(self.config_params['input_size']/16)
        self.last = nn.Linear(dim_flatten, self.num_classes, bias=False) if n_heads is None \
            else nn.ModuleList([nn.Linear(dim_flatten, self.num_classes, bias=False) for _ in range(n_heads)])

        self.initialize_weights()

        self.pred_with_transformer = ['conv3']
        if config_params['predict_bn']:
            self.pred_with_transformer.append('bn3')

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # Apply Xavier initialization to weights of convolutional and linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                # Initialize batch normalization layers with a small variance to avoid scale issues
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, x, task_idx=None):
        x = self.conv1(x)
        if self.config_params['use_bn']:
            x = self.bn1(x)
        x = self.maxpool(self.act(x))
        x = self.conv2(x)
        if self.config_params['use_bn']:
            x = self.bn2(x)
        x = self.maxpool(self.act(x))
        x = self.conv3(x)
        if self.config_params['use_bn']:
            x = self.bn3(x)
        x = self.maxpool(self.act(x))
        x = self.flat(x)
        # account for multi-head approach (for TIL baselines)
        out = self.last(x) if self.n_heads is None else self.last[task_idx](x)
        return out
