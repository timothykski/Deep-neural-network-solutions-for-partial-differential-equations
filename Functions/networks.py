import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class Resnet(nn.Module):
    def __init__(self, layers, stable, activation):
        super(Resnet, self).__init__()
        self.stable = stable
        self.epsilon = 0.01
        self.activation_function = activation

        # Define the layers
        self.input_layer = nn.Linear(layers[0], layers[1])
        self.hidden_layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(1, len(layers) - 2)])
        self.output_layer = nn.Linear(layers[-2], layers[-1])

        # If NAIS-Net, add additional input layers
        if self.stable:
            self.input_layers = nn.ModuleList([nn.Linear(layers[0], layers[i]) for i in range(1, len(layers) - 1)])

    def stable_forward(self, layer, out):  # Building block for the NAIS-Net
        weights = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(weights.t(), weights)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** (1 / 2) * RtR / (norm ** (1 / 2))
        A = RtR + torch.eye(RtR.shape[0]).to(out.device) * self.epsilon
        return F.linear(out, -A, layer.bias)

    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation_function(out)
        u = x

        for i, layer in enumerate(self.hidden_layers):
            shortcut = out.clone()
            if self.stable:
                out = self.stable_forward(layer, out)
                out = out + self.input_layers[i](u)
            else:
                out = layer(out)
            out = self.activation_function(out)
            out = out + shortcut

        out = self.output_layer(out)
        return out



class SDEnet(nn.Module):

    def __init__(self, layers, activation):
        super(SDEnet, self).__init__()

        self.layers = nn.ModuleList()
        self.brownian = nn.ModuleList()

        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
            if i > 0 and i < len(layers) - 2:
                self.brownian.append(nn.Linear(in_features=layers[i], out_features=1, bias=False))

        self.activation = activation
        self.epsilon = 1e-4
        self.h = 0.1

    def product(self, layer, out):
        weights = layer.weight
        RtR = torch.matmul(weights.t(), weights)
        A = RtR + torch.eye(RtR.shape[0]).cuda() * self.epsilon

        return F.linear(out, A, layer.bias)

    def forward(self, x):
        out = self.layers[0](x)
        out = self.activation(out)

        for i, layer in enumerate(self.layers[1:-1]):
            shortcut = out
            out = layer(out)
            out = shortcut + self.h * self.activation(out) + self.h ** (1 / 2) * self.product(self.brownian[i],
                                                                                              torch.rand_like(out))
            # out = shortcut + self.activation(out) + 0.4*torch.ones_like(out)*torch.rand_like(out)

        out = self.layers[-1](out)

        return out


class VerletNet(nn.Module):

    def __init__(self, layers, activation):
        super(VerletNet, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))

        self.h = 0.5
        self.activation = activation

    def transpose(self, layer, out):

        return F.linear(out, layer.weight.t(), layer.bias)

    def forward(self, x):

        out = self.layers[0](x)
        out = self.activation(out)

        z = torch.zeros_like(out)

        for layer in self.layers[1:-1]:
            shortcut = out
            out = self.transpose(layer, out)
            z = z - self.activation(out)
            out = layer(z)
            out = shortcut + self.activation(out)

        out = self.layers[-1](out)

        return out