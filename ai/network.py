import torch
import torch.nn as nn
import numpy as np

class SnakeNet(nn.Module):
    def __init__(self, layer_sizes=None):
        super().__init__()
        if layer_sizes is None:
            layer_sizes = [32, 20, 12, 3]

        self.layer_sizes = layer_sizes
        layers = []

        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def predict_action(self, features):
        with torch.no_grad():
            tensor_x = torch.from_numpy(features).float().unsqueeze(0)
            logits = self.forward(tensor_x)
            action = torch.argmax(logits, dim=1).item()
            return action

    def get_weights_vector(self):
        parts = [param.data.cpu().numpy().flatten() for param in self.parameters()]
        return np.concatenate(parts).astype(np.float32)

    def set_weights_vector(self, weights_vector):
        idx = 0
        with torch.no_grad():
            for param in self.parameters():
                numel = param.numel()
                chunk = weights_vector[idx : idx + numel]
                param.data.copy_(torch.from_numpy(chunk).view(param.shape))
                idx += numel

    @property
    def total_params(self):
        return sum(param.numel() for param in self.parameters())
