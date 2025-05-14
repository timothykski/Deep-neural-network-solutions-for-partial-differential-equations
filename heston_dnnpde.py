import os
import sys
import time
import tqdm
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import integrate
import math


class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module"""

    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Naisnet(nn.Module):

    def __init__(self, layers, stable, activation):
        super(Naisnet, self).__init__()

        self.layers = layers
        self.layer1 = nn.Linear(in_features=layers[0], out_features=layers[1])
        self.layer2 = nn.Linear(in_features=layers[1], out_features=layers[2])
        self.layer2_input = nn.Linear(in_features=layers[0], out_features=layers[2])
        self.layer3 = nn.Linear(in_features=layers[2], out_features=layers[3])
        if len(layers) == 5:
            self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
            self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
        elif len(layers) == 6:
            self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
            self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
            self.layer4_input = nn.Linear(in_features=layers[0], out_features=layers[4])
            self.layer5 = nn.Linear(in_features=layers[4], out_features=layers[5])

        self.activation = activation

        self.epsilon = 0.01
        self.stable = stable

    def project(self, layer, out):  # Building block for the NAIS-Net
        weights = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(weights.t(), weights)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** (1 / 2) * RtR / (norm ** (1 / 2))
        # A = RtR + torch.eye(RtR.shape[0]).cuda() * self.epsilon
        A = RtR + torch.eye(RtR.shape[0]) * self.epsilon
        return F.linear(out, -A, layer.bias)

    def forward(self, x):
        u = x

        out = self.layer1(x)
        out = self.activation(out)

        shortcut = out
        if self.stable:
            out = self.project(self.layer2, out)
            out = out + self.layer2_input(u)
        else:
            out = self.layer2(out)
        out = self.activation(out)
        out = out + shortcut

        if len(self.layers) == 4:
            out = self.layer3(out)
            return out

        if len(self.layers) == 5:
            shortcut = out
            if self.stable:
                out = self.project(self.layer3, out)
                out = out + self.layer3_input(u)
            else:
                out = self.layer3(out)
            out = self.activation(out)
            out = out + shortcut

            out = self.layer4(out)
            return out

        if len(self.layers) == 6:
            shortcut = out
            if self.stable:
                out = self.project(self.layer3, out)
                out = out + self.layer3_input(u)
            else:
                out = self.layer3(out)
            out = self.activation(out)
            out = out + shortcut

            shortcut = out
            if self.stable:
                out = self.project(self.layer4, out)
                out = out + self.layer4_input(u)
            else:
                out = self.layer4(out)

            out = self.activation(out)
            out = out + shortcut

            out = self.layer5(out)
            return out

        return out


class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation, correlation_type="no_correlation"):
        # Check if CUDA is available and set the appropriate device (GPU or CPU)
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        # Initialize the initial condition, convert it to a PyTorch tensor, and send to the device
        # self.Xi = torch.from_numpy(Xi).float().unsqueeze(0).repeat(self.M, 1).to(self.device)
        self.Xi = torch.from_numpy(Xi).float().to(self.device)
        self.Xi.requires_grad = True

        # Store other parameters as attributes of the class.
        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.Mm = Mm  # number of discretization points for the SDE
        self.strike = 1.0  # strike price

        self.mode = mode  # architecture of the neural network
        self.activation = activation  # activation function        # Initialize the activation function based on the provided parameter
        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()
        elif activation == "Tanh":
            self.activation_function = nn.Tanh()

        # Initialize the neural network based on the chosen mode

        if self.mode == "FC":
            # Fully Connected architecture
            self.layers = []
            for i in range(len(layers) - 2):
                self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
                self.layers.append(self.activation_function)
            self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))
            self.model = nn.Sequential(*self.layers).to(self.device)

        elif self.mode == "Naisnet":
            # NAIS-Net architecture
            self.model = Naisnet(layers, stable=True, activation=self.activation_function).to(self.device)

        # Apply a custom weights initialization to the model.
        self.model.apply(self.weights_init)

        # Initialize lists to record training loss and iterations.
        self.training_loss = []
        self.iteration = []
        self.Y0_values = []
        self.correlation_type = correlation_type
        self.correlation_matrix = self.generate_correlation_matrix(D)

    def generate_correlation_matrix(self, D):
        if self.correlation_type == "no_correlation":
            return np.eye(D)
        elif self.correlation_type == "random_correlation":
            return self.generate_random_correlation_matrix(D)
        elif self.correlation_type == "restricted_random_correlation":
            return self.generate_random_correlation_matrix(D, restrict_positive=True)
        else:
            raise ValueError("Invalid correlation type")

    def generate_random_correlation_matrix(self, D, restrict_positive=False):
        random_matrix = np.random.randn(D, D)
        if restrict_positive:
            random_matrix = np.abs(random_matrix)
        random_corr_matrix = np.dot(random_matrix, random_matrix.T)
        np.fill_diagonal(random_corr_matrix, 1)
        d = np.sqrt(np.diag(random_corr_matrix))
        random_corr_matrix = random_corr_matrix / np.outer(d, d)
        return self._make_positive_definite(random_corr_matrix)

    def _make_positive_definite(self, matrix):
        epsilon = 1e-6
        while not np.all(np.linalg.eigvals(matrix) > 0):
            matrix += epsilon * np.eye(matrix.shape[0])
            epsilon *= 2
        return matrix


    def weights_init(self, m):
        # Custom weight initialization method for neural network layers
        # Parameters:
        # m: A layer of the neural network

        if type(m) == nn.Linear:
            # Initialize the weights of the linear layer using Xavier uniform initialization
            torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):  # M x 1, M x D
        # Debug print
        # print(f"Initial shapes - t: {t.shape}, X: {X.shape}")

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if X.dim() == 1:
            X = X.unsqueeze(-1)

        # Debug print
        # print(f"Before cat - t: {t.shape}, X: {X.shape}")

        # Concatenate the time and state variables along second dimension
        input = torch.cat((t, X), 1)

        # Debug print
        # print(f"After cat - input: {input.shape}")

        # Pass the concatenated input through the neural network model
        u = self.model(input)  # M x 1


        # Compute the gradient of the output u with respect to the state variables X
        Du = torch.autograd.grad(
            outputs=u,
            inputs=X,
            grad_outputs=torch.ones_like(u),
            allow_unused=True,
            retain_graph=True,
            create_graph=True)[0]

        return u, Du

    def Dg_tf(self, X):  # M x D
        # Calculates the gradient of the function g with respect to the input X
        # Parameters:
        # X: A batch of state variables, with dimensions M x D

        g = self.g_tf(X)  # M x 1

        # Now, compute the gradient of g with respect to X
        # The gradient is calculated for each input in the batch, resulting in a tensor of dimensions M x D
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g),
                                 allow_unused=True, retain_graph=True, create_graph=True)[0]

        return Dg

    def loss_function(self, t, W, Xi):
        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        # Adjust Xi to match the batch size

        if Xi.shape[0] == 1:
            X0 = Xi.view(1, self.D).repeat(self.M, 1)
        else:
            X0 = Xi.view(self.M, self.D)

        Y0, Z0 = self.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)

            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
                keepdim=True)

            Y1, Z1 = self.net_u(t1, X1)
            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1

            X_list.append(X0)
            Y_list.append(Y0)

        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):  # Generate time + a Brownian motion
        # Generates a minibatch of time steps and corresponding Brownian motion paths

        T = self.T  # Terminal time
        M = self.M  # Number of trajectories (batch size)
        N = self.N  # Number of time snapshots
        D = self.D  # Number of dimensions

        # Initialize arrays for time steps and Brownian increments
        Dt = np.zeros((M, N + 1, 1))  # Time step sizes for each trajectory and time snapshot
        DW = np.zeros((M, N + 1, D))  # Brownian increments for each trajectory, time snapshot, and dimension

        # Calculate the time step size
        dt = T / N

        # Populate the time step sizes for each trajectory and time snapshot (excluding the initial time)
        Dt[:, 1:, :] = dt
        # Generate Brownian increments for each trajectory and time snapshot
        DW_uncorrelated = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        # Apply correlation using Cholesky decomposition
        L = np.linalg.cholesky(self.correlation_matrix)
        DW[:, 1:, :] = np.einsum('ij,mnj->mni', L, DW_uncorrelated)


        # Cumulatively sum the time steps and Brownian increments to get the actual time values and Brownian paths
        t = np.cumsum(Dt, axis=1)  # Cumulative time for each trajectory and time snapshot
        W = np.cumsum(DW, axis=1)  # Cumulative Brownian motion for each trajectory, time snapshot, and dimension

        # Convert the numpy arrays to PyTorch tensors and transfer them to the configured device (CPU or GPU)
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        # Return the time values and Brownian paths.
        return t, W

    def train(self, N_Iter, learning_rate, optimizer_type='Adam'):
        # Train the neural network model.
        # Parameters:
        # N_Iter: Number of iterations for the training process
        # learning_rate: Learning rate for the optimizer

        # Initialize an array to store temporary loss values for averaging
        loss_temp = np.array([])

        # Check if there are previous iterations and set the starting iteration number
        previous_it = 0
        if self.iteration != []:
            previous_it = self.iteration[-1]

        # Set up the optimizer for the neural network with the specified learning rate
        if optimizer_type == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'Adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'Adamax':
            self.optimizer = optim.Adamax(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'ASGD':
            self.optimizer = optim.ASGD(self.model.parameters(), lr=learning_rate)
        elif optimizer_type == 'LBFGS':
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Optimizer type '{optimizer_type}' is not recognized.")

        # Record the start time for timing the training process
        start_time = time.time()
        cumulative_time = 0  # Track cumulative time
        time_logs = [] # List to track cumulative time at each iteration
        min_loss = float('inf')
        min_loss_state = None

        def closure():
            self.optimizer.zero_grad()
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            loss.backward()
            return loss

        # Training loop
        for it in range(previous_it, previous_it + N_Iter):
            if it >= 4000 and it < 20000:
                self.N = int(np.ceil(self.Mm ** (int(it / 4000) + 1)))
            elif it < 4000:
                self.N = int(np.ceil(self.Mm))

            # Zero the gradients before each iteration
            self.optimizer.zero_grad()

            # Fetch a minibatch of time steps and Brownian motion paths
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D

            # Compute the loss for the current batch
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            if torch.isnan(loss):
                print(f"NaN loss detected at iteration {it}. Skipping this iteration")
                continue

            loss.backward()  # Compute the gradients of the loss w.r.t. the network parameters

            if optimizer_type == 'LBFGS':
                self.optimizer.step(closure)
            else:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()  # Update the network parameters based on the gradients

            # Store the current loss value for later averaging
            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            # Track the minimum loss and its corresponding state
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_loss_state = (X_pred.clone().detach(), Y_pred.clone().detach())

            # Print the training progress every 100 iterations
            if it % 100 == 0:
                elapsed = time.time() - start_time  # Calculate the elapsed time
                cumulative_time += elapsed  # Update cumulative time
                time_logs.append(cumulative_time)  # Log the cumulative time
                # print(
                #     f'It: {it}, Loss: {loss:.3e}, Y0: {Y0_pred:.3f}, Time: {elapsed:.2f}, Learning Rate: {learning_rate:.3e}')
                start_time = time.time()  # Reset the start time for the next print interval

            # Record the average loss and iteration number every 100 iterations
            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())  # Append the average loss
                loss_temp = np.array([])  # Reset the temporary loss array
                self.iteration.append(it)  # Append the current iteration number
                self.Y0_values.append(Y0_pred.item())

        # Stack the iteration and training loss for plotting
        # graph = np.stack((self.iteration, self.training_loss))
        graph = np.column_stack((self.iteration, self.training_loss, self.Y0_values))

        return graph

    def predict(self, Xi_star, t_star, W_star):
        # print(f"Predict input shapes - Xi_star: {Xi_star.shape}, t_star: {t_star.shape}, W_star: {W_star.shape}")

        # Ensure Xi_star is a tensor and has the correct shape
        if not isinstance(Xi_star, torch.Tensor):
            Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star = Xi_star.view(-1, self.D)  # Reshape to (batch_size, D)
        Xi_star.requires_grad = True

        # Ensure t_star and W_star are tensors
        if not isinstance(t_star, torch.Tensor):
            t_star = torch.from_numpy(t_star).float().to(self.device)
        if not isinstance(W_star, torch.Tensor):
            W_star = torch.from_numpy(W_star).float().to(self.device)

        # Adjust batch sizes
        batch_size = max(Xi_star.shape[0], t_star.shape[0], W_star.shape[0])
        self.M = batch_size  # Update the batch size

        if Xi_star.shape[0] == 1:
            Xi_star = Xi_star.repeat(batch_size, 1)
        if t_star.shape[0] == 1:
            t_star = t_star.repeat(batch_size, 1, 1)
        if W_star.shape[0] == 1:
            W_star = W_star.repeat(batch_size, 1, 1)

        # print(f"Before loss_function - Xi_star: {Xi_star.shape}, t_star: {t_star.shape}, W_star: {W_star.shape}")

        # Compute the loss and obtain predicted states (X_star) and outputs (Y_star) using the trained model
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star  # Return only the first time step of Y_star

    def save_model(self, file_name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_loss': self.training_loss,
            'iteration': self.iteration
        }, file_name)

    def load_model(self, file_name):
        checkpoint = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_loss = checkpoint['training_loss']
        self.iteration = checkpoint['iteration']

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        pass

    @abstractmethod
    def g_tf(self, X):  # M x D
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D


class HestonFBSNN(FBSNN):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation, correlation_type="no_correlation", kappa=2.0,
                 theta=0.2, sigma=0.3, rho=0.8, v0=0.2, payoff_type = 'discontinuous'):
        super().__init__(Xi, T, M, N, 1, Mm, layers, mode, activation, correlation_type)

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.payoff_type = payoff_type  # Add payoff type attribute
        self.y0_values = [] #todo: currently this is not even used --> ensure the correct use of this

        # Modify the input layer of the neural network to account for both S and v
        if self.mode == "FC":
            self.model[0] = nn.Linear(in_features=3, out_features=layers[1])  # 3 inputs: t, S, v
        elif self.mode == "Naisnet":
            self.model.layer1 = nn.Linear(in_features=3, out_features=layers[1])
            self.model.layer2_input = nn.Linear(in_features=3, out_features=layers[2])
            if len(layers) >= 5:
                self.model.layer3_input = nn.Linear(in_features=3, out_features=layers[3])
            if len(layers) == 6:
                self.model.layer4_input = nn.Linear(in_features=3, out_features=layers[4])

        # Initialize the weights
        self.initialize_weights()

    def g_tf(self, X):
        S = X[:, 0:1] if X.dim() > 1 else X
        if self.payoff_type == 'discontinuous':
            # Discontinuous payoff (standard call option)
            return torch.maximum(S - self.strike, torch.tensor(0.0).to(self.device))
        elif self.payoff_type == 'continuous':
            # Continuous/smoothed payoff using sigmoid
            K = self.strike
            alpha = 10.0  # Adjust the alpha value as needed for smoothness
            smoothed_payoff = (S - K) / (1 + torch.exp(-alpha * (S - K)))
            return smoothed_payoff.to(self.device)
        else:
            raise ValueError("Invalid payoff type. Choose 'discontinuous' or 'continuous'.")

    def net_u(self, t, X):
        # Handling batch sizes properly for Heston case
        if X.shape[1] == 2:  # Heston case
            S, v = X[:, 0:1], X[:, 1:2]
            if t.dim() == 1:
                t = t.unsqueeze(-1)
            input = torch.cat((t, S, v), dim=1)  # Concatenates time, S, and v
            u = self.model(input)
            u = torch.clamp(u, min=0.0)  # Clamp to ensure non-negativity

            Du = torch.autograd.grad(
                outputs=u,
                inputs=(S, v),
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True
            )
            return u, Du[0], Du[1]  # u, dU/dS, dU/dv
        else:
            return super().net_u(t, X)
    def initialize_weights(self):
        for param in self.model.parameters():
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param, gain=0.5)
            else:
                torch.nn.init.zeros_(param)

    def mu_tf(self, t, X, Y=None, Z=None):
        S, v = X[:, 0:1], X[:, 1:2]
        mu_S = 0.05 * S
        mu_v = self.kappa * (self.theta - v)
        return torch.cat([mu_S, mu_v], dim=1).clamp(-100, 100)

    def sigma_tf(self, t, X, Y=None):
        S, v = X[:, 0:1], X[:, 1:2]
        M = S.shape[0]
        sigma_S = torch.sqrt(torch.clamp(v, min=1e-8)) * S
        sigma_v = self.sigma * torch.sqrt(torch.clamp(v, min=1e-8))

        diffusion_matrix = torch.zeros((M, 2, 2)).to(self.device)
        diffusion_matrix[:, 0, 0] = sigma_S.squeeze()
        diffusion_matrix[:, 1, 1] = sigma_v.squeeze()
        diffusion_matrix[:, 0, 1] = self.rho * sigma_v.squeeze()
        diffusion_matrix[:, 1, 0] = self.rho * sigma_S.squeeze()

        return diffusion_matrix.clamp(-100, 100)

    def phi_tf(self, t, X, Y, Z):
        r = 0.05  # Risk-free rate
        return r * Y

    def loss_function(self, t, W, Xi):
        loss = 0
        X_list, Y_list = [], []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        # S0 = Xi[:, 0:1]
        S0 = Xi[:, 0:1].repeat(self.M, 1) if Xi.shape[0] == 1 else Xi[:, 0:1]
        v0 = torch.full((self.M, 1), self.v0).to(self.device)


        X0 = torch.cat([S0, v0], dim=1)
        Y0, Z_S0, Z_v0 = self.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            dW = W1 - W0
            mu = self.mu_tf(t0, X0)
            sigma = self.sigma_tf(t0, X0)

            # Handle batch sizes in matrix multiplication
            X1 = X0 + mu * (t1 - t0) + torch.einsum('mij,mj->mi', sigma, dW)

            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, (Z_S0, Z_v0)) * (t1 - t0) + \
                       torch.sum(Z_S0 * torch.sum(sigma[:, 0, :] * dW, dim=1, keepdim=True) + \
                                 Z_v0 * torch.sum(sigma[:, 1, :] * dW, dim=1, keepdim=True), dim=1, keepdim=True)

            Y1, Z_S1, Z_v1 = self.net_u(t1, X1)

            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            t0, W0, X0, Y0, Z_S0, Z_v0 = t1, W1, X1, Y1, Z_S1, Z_v1
            X_list.append(X0)
            Y_list.append(Y0)

        # Terminal conditions handling for any batch size
        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z_S1 - self.Dg_tf(X1[:, 0:1]), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0]

    def predict(self, Xi_star, t_star, W_star):
        if not isinstance(Xi_star, torch.Tensor):
            Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        else:
            Xi_star = Xi_star.to(self.device)

        # Ensure Xi_star has two columns (S and v)
        if Xi_star.shape[1] == 1:
            v0 = torch.full((Xi_star.shape[0], 1), self.v0).to(self.device)
            Xi_star = torch.cat([Xi_star, v0], dim=1)

        Xi_star.requires_grad = True

        if not isinstance(t_star, torch.Tensor):
            t_star = torch.from_numpy(t_star).float().to(self.device)
        if not isinstance(W_star, torch.Tensor):
            W_star = torch.from_numpy(W_star).float().to(self.device)

        batch_size = max(Xi_star.shape[0], t_star.shape[0], W_star.shape[0])
        self.M = batch_size

        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
        return X_star[:, :, 0:1], X_star[:, :, 1:2], Y_star  # S, v, Y

    def calculate_greeks(self, S, v, t):
        S = torch.tensor(S, dtype=torch.float32, requires_grad=True, device=self.device)
        v = torch.tensor(v, dtype=torch.float32, requires_grad=True, device=self.device)
        t = torch.tensor(t, dtype=torch.float32, device=self.device)

        # Stack S and v into the input X, accommodating different batch sizes
        X = torch.stack([S, v], dim=-1)

        # Call net_u and get Y, dU/dS (delta), and dU/dv
        Y, delta, _ = self.net_u(t.unsqueeze(-1), X)

        # Compute the second-order derivative to obtain gamma
        gamma = torch.autograd.grad(delta, S, grad_outputs=torch.ones_like(delta), create_graph=True)[0]

        return Y.detach().cpu().numpy(), delta.detach().cpu().numpy(), gamma.detach().cpu().numpy()


class PredictionGenerator:
    def __init__(self, model, Xi, num_samples):
        self.model = model
        self.Xi = Xi
        self.num_samples = num_samples

    def generate_predictions(self):
        np.random.seed(42)
        t_test, W_test = self.model.fetch_minibatch()
        X_pred, Y_pred = self.model.predict(self.Xi, t_test, W_test)

        if type(t_test).__module__ != 'numpy':
            t_test = t_test.cpu().numpy()
        if type(X_pred).__module__ != 'numpy':
            X_pred = X_pred.cpu().detach().numpy()
        if type(Y_pred).__module__ != 'numpy':
            Y_pred = Y_pred.cpu().detach().numpy()

        X_pred_all = [X_pred]
        Y_pred_all = [Y_pred]
        t_test_all = [t_test]

        for _ in range(self.num_samples - 1):
            t_test_i, W_test_i = self.model.fetch_minibatch()
            X_pred_i, Y_pred_i = self.model.predict(self.Xi, t_test_i, W_test_i)
            if type(X_pred_i).__module__ != 'numpy':
                X_pred_i = X_pred_i.cpu().detach().numpy()
            if type(Y_pred_i).__module__ != 'numpy':
                Y_pred_i = Y_pred_i.cpu().detach().numpy()
            if type(t_test_i).__module__ != 'numpy':
                t_test_i = t_test_i.cpu().numpy()

            t_test_all.append(t_test_i)
            X_pred_all.append(X_pred_i)
            Y_pred_all.append(Y_pred_i)

        t_test = np.concatenate(t_test_all, axis=0)  # Here is the problem, the size of t_test changed
        X_pred = np.concatenate(X_pred_all, axis=0)
        Y_pred = np.concatenate(Y_pred_all, axis=0)
        # X_pred = X_pred[:500, :]

        return t_test, W_test, X_pred, Y_pred

class HestonOptionPriceCalculator:
    def __init__(self, S0, K, r, T, kappa, theta, sigma, rho, v0):
        # Initialize the HestonClosedFormSurface instance with the initial parameters
        self.heston_model = HestonClosedFormSurface(S0, K, r, T, kappa, theta, sigma, rho, v0)

    def calculate_call_option_prices(self, S_pred, v_pred, time_array):
        rows, cols = S_pred.shape

        option_prices = np.zeros((rows, cols))
        deltas = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                S = S_pred[i, j]
                v = v_pred[i, j]

                # Calculate the time to maturity for the current time step
                t = time_array[min(j, len(time_array) - 1)]
                time_to_maturity = T - t

                if np.all(time_to_maturity) > 0:
                    option_prices[i, j] = self.heston_model.call_price(S, v)
                    delta_grid = self.heston_model.delta_surface([S], [v])
                    deltas[i, j] = delta_grid[0, 0]  # Extract the delta value for S and v
                else:
                    # Handle the payoff at maturity
                    option_prices[i, j] = max(S - self.heston_model.K, 0)
                    deltas[i, j] = 1 if S > self.heston_model.K else (0.5 if S == self.heston_model.K else 0)

        return option_prices, deltas



class HestonPredictionGenerator:
    def __init__(self, model, Xi, num_samples):
        self.model = model
        self.Xi = Xi
        self.num_samples = num_samples

    def generate_predictions(self):
        np.random.seed(42)
        t_test, W_test = self.model.fetch_minibatch()

        # Ensure Xi is a torch tensor and has the correct shape
        if not isinstance(self.Xi, torch.Tensor):
            Xi = torch.from_numpy(self.Xi).float().to(self.model.device)
        else:
            Xi = self.Xi.to(self.model.device)

        # Add volatility column if needed
        if Xi.shape[1] == 1:
            v0 = torch.full((Xi.shape[0], 1), self.model.v0).to(self.model.device)
            Xi = torch.cat([Xi, v0], dim=1)

        S_pred, v_pred, Y_pred = self.model.predict(Xi, t_test, W_test)

        t_test = t_test.cpu().numpy()
        S_pred = S_pred.cpu().detach().numpy()
        v_pred = v_pred.cpu().detach().numpy()
        Y_pred = Y_pred.cpu().detach().numpy()

        S_pred_all, v_pred_all, Y_pred_all, t_test_all = [S_pred], [v_pred], [Y_pred], [t_test]

        for _ in range(self.num_samples - 1):
            t_test_i, W_test_i = self.model.fetch_minibatch()
            S_pred_i, v_pred_i, Y_pred_i = self.model.predict(Xi, t_test_i, W_test_i)

            t_test_i = t_test_i.cpu().numpy()
            S_pred_i = S_pred_i.cpu().detach().numpy()
            v_pred_i = v_pred_i.cpu().detach().numpy()
            Y_pred_i = Y_pred_i.cpu().detach().numpy()

            t_test_all.append(t_test_i)
            S_pred_all.append(S_pred_i)
            v_pred_all.append(v_pred_i)
            Y_pred_all.append(Y_pred_i)

        t_test = np.concatenate(t_test_all, axis=0)
        S_pred = np.concatenate(S_pred_all, axis=0)
        v_pred = np.concatenate(v_pred_all, axis=0)
        Y_pred = np.concatenate(Y_pred_all, axis=0)

        return t_test, W_test, S_pred, v_pred, Y_pred



class HestonClosedFormSurface:
    def __init__(self, S0, K, r, T, kappa, theta, sigma, rho, v0):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.lambda_ = 0  # lambda is set to zero as per the typical assumption

    def char_func(self, phi, S, V, P):
        a = self.kappa * self.theta
        u = 0.5
        b = self.kappa + self.lambda_

        if P == 1:
            u = 0.5
            b = self.kappa + self.lambda_ - self.rho * self.sigma
        elif P == 2:
            u = -0.5
            b = self.kappa + self.lambda_

        d = np.sqrt((self.rho * self.sigma * 1j * phi - b) ** 2 - (self.sigma ** 2) * (2 * u * 1j * phi - phi ** 2))
        g = (b - self.rho * self.sigma * 1j * phi + d) / (b - self.rho * self.sigma * 1j * phi - d)

        # Avoid division by zero in the calculation of C and D
        g_exp_dT = np.where(np.abs(g * np.exp(d * self.T) - 1) < 1e-8, 1e-8, g * np.exp(d * self.T))

        C = self.r * 1j * phi * self.T + (a / (self.sigma ** 2)) * (
                (b - self.rho * self.sigma * 1j * phi + d) * self.T - 2 * np.log((1 - g_exp_dT) / (1 - g)))
        D = ((b - self.rho * self.sigma * 1j * phi + d) / (self.sigma ** 2)) * (
                    (1 - np.exp(d * self.T)) / (1 - g_exp_dT))

        # Adding a small epsilon to S to avoid taking log of 0
        S = np.maximum(S, 1e-8)
        return np.exp(C + D * V + 1j * phi * np.log(S))

    def integrand(self, phi, P, S, V):
        return np.real((np.exp(-1j * phi * np.log(self.K)) * self.char_func(phi, S, V, P)) / (1j * phi + 1e-10))

    def P1(self, S, V):
        integral = \
        integrate.quad(lambda phi: self.integrand(phi, 1, S, V), 0, 100, limit=100, epsabs=1e-8, epsrel=1e-8)[0]
        return 0.5 + (1 / np.pi) * integral

    def P2(self, S, V):
        integral = \
        integrate.quad(lambda phi: self.integrand(phi, 2, S, V), 0, 100, limit=100, epsabs=1e-8, epsrel=1e-8)[0]
        return 0.5 + (1 / np.pi) * integral

    def call_price(self, S, V):
        P1 = self.P1(S, V)
        P2 = self.P2(S, V)
        return np.exp(-self.r * self.T) * (S * P1 - self.K * P2)

    def price_surface(self, S_values, V_values):
        price_grid = np.zeros((len(S_values), len(V_values)))

        for i, S in enumerate(S_values):
            for j, V in enumerate(V_values):
                price_grid[i, j] = self.call_price(S, V)

        return price_grid

    def delta_surface(self, S_values, V_values):
        delta_grid = np.zeros((len(S_values), len(V_values)))
        dS = S_values[1] - S_values[0]

        for i in range(len(S_values) - 1):
            for j in range(len(V_values)):
                delta_grid[i, j] = (self.call_price(S_values[i + 1], V_values[j]) - self.call_price(S_values[i],
                                                                                                    V_values[j])) / dS

        return delta_grid

    def gamma_surface(self, S_values, V_values):
        gamma_grid = np.zeros((len(S_values), len(V_values)))
        dS = S_values[1] - S_values[0]

        for i in range(1, len(S_values) - 1):
            for j in range(len(V_values)):
                gamma_grid[i, j] = (self.call_price(S_values[i + 1], V_values[j]) - 2 * self.call_price(S_values[i],
                                                                                                        V_values[
                                                                                                            j]) + self.call_price(
                    S_values[i - 1], V_values[j])) / (dS ** 2)

        return gamma_grid

    def plot_surfaces(self, S_values, V_values):
        option_price_grid = self.price_surface(S_values, V_values)
        delta_grid = self.delta_surface(S_values, V_values)
        gamma_grid = self.gamma_surface(S_values, V_values)

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={'projection': '3d'})
        S, V = np.meshgrid(S_values, V_values)

        axs[0].plot_surface(S, V, option_price_grid.T, cmap='viridis')
        axs[0].set_title('Option Price Surface')
        axs[0].set_xlabel('S')
        axs[0].set_ylabel('V')
        axs[0].set_zlabel('Price')
        axs[0].view_init(elev=30, azim=130)  # Adjust the angle here

        axs[1].plot_surface(S, V, delta_grid.T, cmap='viridis')
        axs[1].set_title('Delta (∂Price/∂S)')
        axs[1].set_xlabel('S')
        axs[1].set_ylabel('V')
        axs[1].set_zlabel('Delta')
        axs[1].view_init(elev=30, azim=130)  # Adjust the angle here

        axs[2].plot_surface(S, V, gamma_grid.T, cmap='viridis')
        axs[2].set_title('Gamma (∂²Price/∂S²)')
        axs[2].set_xlabel('S')
        axs[2].set_ylabel('V')
        axs[2].set_zlabel('Gamma')
        axs[2].view_init(elev=30, azim=130)  # Adjust the angle here

        # plt.show()



class TrainingPhases:
    def __init__(self, model):
        self.model = model

    def train_initial_phase(self, n_iter, lr, optimizer_type='Adam'):
        print("Starting initial training phase...")
        tot = time.time()
        print(self.model.device)
        graph = self.model.train(n_iter, lr, optimizer_type)
        print("Initial training phase completed. Total time:", time.time() - tot, "s")
        return graph

    def fine_tuning_phase(self, n_iter, lr, optimizer_type='Adam'):
        print("Starting fine-tuning phase...")
        tot = time.time()
        print(self.model.device)
        graph = self.model.train(n_iter, lr, optimizer_type)
        print("Fine-tuning phase completed. Total time:", time.time() - tot, "s")
        return graph



class TrainingPlot:
    def __init__(self, save_path):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def figsize(self, scale, nplots=1):
        # todo: please double check this plot scaling function
        fig_width_pt = 438.17227
        inches_per_pt = 1.0 / 72.27
        golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
        fig_width = fig_width_pt * inches_per_pt * scale
        fig_height = nplots * fig_width * golden_mean
        fig_size = [fig_width, fig_height]
        return fig_size

    def plot_training_loss(self, graph, D, mode, activation, optimizer, M, payoff_type):
        plt.figure(figsize=self.figsize(1.0))
        plt.plot(graph[0], graph[1])
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.yscale("log")
        plt.title(f'Training Loss (M={M}, Payoff={payoff_type}, {mode}-{activation}-{optimizer})')
        filename = f'CallOption{D}DLoss_M{M}_{payoff_type}_{mode}_{activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()
        # plt.show()

    def plot_l2_error(self, iterations, l2_errors, D, mode, activation, optimizer, M, payoff_type):
        # l2_errors = np.sqrt((Y_true - graph[:, 2])**2)  # Y_true - Y0 values

        plt.figure(figsize=self.figsize(1.0))
        plt.plot(iterations, l2_errors)  # Iteration vs L2 Error
        plt.xlabel('Iterations')
        plt.ylabel('L2 Error')
        plt.yscale("log")
        plt.title(f'L2 Error (M={M}, Payoff={payoff_type}, {mode}-{activation}-{optimizer})')
        filename = f'CallOption{D}DL2Error_M{M}_{payoff_type}_{mode}_{activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()
        # plt.show()

        return l2_errors

    def plot_y0_evolution(self, iterations, Y0_values, D, mode, activation, optimizer):
        plt.figure(figsize=self.figsize(1.0))
        plt.plot(iterations, Y0_values)  # Iteration vs Y0
        plt.xlabel('Iterations')
        plt.ylabel('Y0')
        plt.title(f'Evolution of Y prediction ({mode}-{activation}-{optimizer})')
        filename = f'CallOption{D}DY0Evolution_{mode}_{activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()

    def plot_prediction(self, t_test, Y_pred, D, model, optimizer, M, payoff_type):
        samples = min(5, Y_pred.shape[0])
        plt.figure(figsize=self.figsize(1.0))
        for i in range(samples):
            plt.plot(t_test[i, :, 0], Y_pred[i, :], label=f'Sample {i+1}' if i == 0 else "")
        plt.xlabel('$t$')
        plt.ylabel('$Y_t = u(t,X_t)$')
        plt.title(
            f'{D}-dimensional Call Option (M={M}, Payoff={payoff_type}, {model.mode}-{model.activation}-{optimizer})')
        plt.legend()
        filename = f'CallOption{D}D_Prediction_M{M}_{payoff_type}_{model.mode}_{model.activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        plt.savefig(os.path.join(self.save_path, filename))
        plt.close()
        # plt.show()

    def plot_exact_vs_learned(self, t_test, Y_pred, Y_test, D, model, optimizer,  M, payoff_type):
        plt.figure(figsize=self.figsize(1.0))
        samples = min(7, Y_pred.shape[0])
        for i in range(samples):
            plt.plot(t_test[i, :, 0], Y_pred[i, :] * 100, 'b', label='Learned $u(t,X_t)$' if i == 0 else "")
            plt.plot(t_test[i, :, 0], Y_test[i, :] * 100, 'r--', label='Exact $u(t,X_t)$' if i == 0 else "")
            plt.plot(t_test[i, -1, 0], Y_test[i, -1] * 100, 'ko', label='$Y_T = u(T,X_T)$' if i == 0 else "")
            plt.plot(t_test[i, 0, 0], Y_pred[i, 0] * 100, 'ks', label='$Y_0 = u(0,X_0)$' if i == 0 else "")

        plt.title(
            f'{D}-dimensional Basket Option (M={M}, Payoff={payoff_type}, {model.mode}-{model.activation}-{optimizer})')
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('$Y_t = u(t,X_t)$')
        filename = f'CallOption{D}DPreds_M{M}_{payoff_type}_{model.mode}_{model.activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        # plt.show()
        plt.close()

    def plot_heston_predictions(self, model, save_path,  M, payoff_type):
        # Create a grid of S and v values
        S = np.linspace(0.5, 1.5, 50)
        v = np.linspace(0.01, 0.5, 50)
        S_grid, v_grid = np.meshgrid(S, v)

        # Calculate option prices, deltas, and gammas
        t = np.full_like(S_grid, model.T)  # Assuming we're interested in prices at maturity
        Y, delta, gamma = model.calculate_greeks(S_grid.flatten(), v_grid.flatten(), t.flatten())

        # Reshape the results
        Y = Y.reshape(S_grid.shape)
        delta = delta.reshape(S_grid.shape)
        gamma = gamma.reshape(S_grid.shape)

        # Plot Option Price Surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_grid, v_grid, Y, cmap='viridis')
        ax.set_xlabel('Stock Price (S)')
        ax.set_ylabel('Volatility (v)')
        ax.set_zlabel('Option Price')
        ax.set_title(f'Heston Model: Option Price Surface (M={M}, Payoff={payoff_type})')
        ax.view_init(elev = 30, azim = 120) # Change the elev and azim to adjust the viewing angle
        fig.colorbar(surf)
        plt.savefig(os.path.join(save_path, f'heston_option_price_surface_M{M}_{payoff_type}.png'))
        plt.show()
        # plt.close()

        # Plot Delta Surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_grid, v_grid, delta, cmap='viridis')
        ax.set_xlabel('Stock Price (S)')
        ax.set_ylabel('Volatility (v)')
        ax.set_zlabel('Delta')
        ax.set_title(f'Heston Model: Delta Surface (M={M}, Payoff={payoff_type})')
        fig.colorbar(surf)
        ax.view_init(elev=30, azim=120)
        plt.savefig(os.path.join(save_path, f'heston_delta_surface_M{M}_{payoff_type}.png'))
        plt.show()
        # plt.close()

        # Plot Gamma Surface
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(S_grid, v_grid, gamma, cmap='viridis')
        ax.set_xlabel('Stock Price (S)')
        ax.set_ylabel('Volatility (v)')
        ax.set_zlabel('Gamma')
        ax.set_title(f'Heston Model: Gamma Surface (M={M}, Payoff={payoff_type})')
        fig.colorbar(surf)
        plt.savefig(os.path.join(save_path, f'heston_gamma_surface_M{M}_{payoff_type}.png'))
        plt.show()
        # plt.close()


class HestonExecutor:
    def __init__(self, Xi, T, M, N, D, layers_template, mode, activation, optimizer, save_path, initial_lr, fine_tuning_lr, initial_iters, fine_tuning_iters,
                 kappa=2.0, theta=0.2, sigma=0.3, rho=0.8, v0=0.2, payoff_type = 'discontinuous'):
        self.Xi = Xi
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.layers_template = layers_template
        self.mode = mode
        self.activation = activation
        self.optimizer = optimizer
        self.save_path = save_path
        self.initial_lr = initial_lr
        self.fine_tuning_lr = fine_tuning_lr
        self.initial_iters = initial_iters
        self.fine_tuning_iters = fine_tuning_iters

        # Heston parameters
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.payoff_type = payoff_type

        # To store L2 errors over iterations
        self.l2_errors = []

    def execute(self):
        print(f"Running Training: Payoff = {self.payoff_type}, Mode={self.mode}, Activation={self.activation}, "
              f"Optimizer={self.optimizer}, M={self.M}, D={self.D}, "
              f"Initial LR={self.initial_lr}, Fine-Tuning LR={self.fine_tuning_lr}")

        # Define layers and Xi based on D
        layers = [self.D + 1] + self.layers_template + [1]
        Xi = np.array([1.0] * self.D)[None, :]

        # Initialize HestonFBSNN model
        model = HestonFBSNN(Xi, self.T, self.M, self.N, self.D, int(self.N ** (1 / 5)), layers, self.mode, self.activation,
                            kappa=self.kappa, theta=self.theta, sigma=self.sigma, rho=self.rho, v0=self.v0, payoff_type = self.payoff_type)

        # Training phases
        trainer = TrainingPhases(model)
        trainer.train_initial_phase(self.initial_iters, self.initial_lr, self.optimizer)
        trainer.fine_tuning_phase(self.fine_tuning_iters, self.fine_tuning_lr, self.optimizer)

        # Prediction
        predictor = HestonPredictionGenerator(model, Xi, num_samples=16)
        t_test, W_test, S_pred, v_pred, Y_pred = predictor.generate_predictions()

        # Correct option prices using the Heston closed-form solution
        correct_heston_surface = HestonClosedFormSurface(
            S0= self.Xi[0,0],
            K=model.strike,
            r=0.05,
            T=self.T,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            v0=self.v0
        )
        Y_true = correct_heston_surface.call_price(1.0, self.v0)
        l2_errors = np.sqrt((Y_true - np.array(model.Y0_values))**2)

        # Calculate correct prices for each path and time point in S_pred
        subset_size = min(100, S_pred.shape[0])
        indices = np.random.choice(S_pred.shape[0], subset_size, replace=False)

        S_pred_subset = S_pred[indices, :, :]
        correct_prices_subset = np.array([
            [correct_heston_surface.call_price(S, self.v0) for S in S_pred_subset[i, :, 0]]
            for i in range(subset_size)
        ])

        # Plot the comparison
        S_pred_flat = S_pred_subset.flatten()
        correct_prices_flat = correct_prices_subset.flatten()

        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        ax.scatter(S_pred_flat, correct_prices_flat, c='red', label='Correct Prices', s=10)
        Y_pred_subset = Y_pred[indices, :, :].flatten()
        ax.scatter(S_pred_flat, Y_pred_subset, c='blue', label='Predicted Prices', s=10)
        plt.title(f'Comparison of Predicted vs. Correct Option Prices M={self.M}, Payoff: {self.payoff_type}')
        plt.xlabel('Stock Price (S)')
        plt.ylabel('Option Price')
        plt.legend()

        plt.savefig(os.path.join(self.save_path, f'correct_vs_predict_M{self.M}_{self.payoff_type}_payoff.png'))

        plt.close()

        # Plotting training loss and other results
        plotter = TrainingPlot(self.save_path)
        plotter.plot_training_loss((model.iteration, model.training_loss), self.D, self.mode, self.activation,
                                   self.optimizer, self.M, self.payoff_type)
        plotter.plot_l2_error(model.iteration, l2_errors, self.D, self.mode, self.activation, self.optimizer, self.M,
                              self.payoff_type)
        plotter.plot_prediction(t_test, Y_pred, self.D, model, self.optimizer, self.M, self.payoff_type)
        plotter.plot_heston_predictions(model, self.save_path, self.M, self.payoff_type)

if __name__ == "__main__":
    Ms = [2, 10, 50, 128, 500]
    N = 50
    Ds = [1]
    layers_template = 4 * [256]
    Xi = np.array([1.0] * max(Ds))[None, :]
    T = 1.0

    modes = ["Naisnet"]
    activations = ["Sine"]
    optimizers = ["Adam"]
    payoff_types = ['discontinuous']

    save_path = r'C:/Users/aa04947/OneDrive - APG/Desktop/dnnpde_output/exp_sv/nu'

    initial_lrs = [1e-3]
    fine_tuning_lrs = [1e-5]
    initial_iters = [200]
    fine_tuning_iters = [50]

    total_start_time = time.time()

    for M in Ms:  # Loop through different batch sizes
        for payoff_type in payoff_types:
            executor = HestonExecutor(
                Xi=Xi,
                T=T,
                M=M,  # Use the current batch size
                N=N,
                D=Ds[0],
                layers_template=layers_template,
                mode=modes[0],
                activation=activations[0],
                optimizer=optimizers[0],
                save_path=save_path,
                initial_lr=initial_lrs[0],
                fine_tuning_lr=fine_tuning_lrs[0],
                initial_iters=initial_iters[0],
                fine_tuning_iters=fine_tuning_iters[0],
                kappa=2.0,
                theta=0.2,
                sigma=0.3,
                rho=0.8,
                v0=0.2,
                payoff_type=payoff_type
            )

            executor.execute()


    total_run_time = time.time() - total_start_time
    print(f"Total run time for the entire algorithm: {total_run_time:.2f} seconds")







