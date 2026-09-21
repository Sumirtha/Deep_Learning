import torch
import torch.nn as nn

class BatchNorm1dFromScratch(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # Learnable scale (gamma) and shift (beta)
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # Running statistics used during inference
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # Mean and Variance across batch dimension (dim 0)
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            # Update running tracking metrics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.detach()
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize and apply scale/shift
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta


class LayerNormFromScratch(nn.Module):
    def __init__(self, feature_dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(feature_dim))
        self.beta = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, x):
        # Mean and Variance across the feature dimension (last dimension)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_hat + self.beta