import os
import sys
import time
import tqdm
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import pairwise_distances
from scipy.stats import gaussian_kde
from scipy.stats import loguniform, truncnorm, lognorm
from scipy.stats import multivariate_normal as normal
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import queue
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools


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
        # todo: no MLMC

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
        time_logs = []  # List to track cumulative time at each iteration
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
            loss.backward()  # Compute the gradients of the loss w.r.t. the network parameters

            if optimizer_type == 'LBFGS':
                self.optimizer.step(closure)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()  # Update the network parameters based on the gradients

            # Store the current loss value for later averaging
            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            # Track the minimum loss and its corresponding state
            if loss.item() < min_loss:
                min_loss = loss.item()
                min_loss_state = (X_pred.clone().detach(), Y_pred.clone().detach())

            # Print the training progress every 100 iterations
            if it % 500 == 0:
                elapsed = time.time() - start_time  # Calculate the elapsed time
                cumulative_time += elapsed  # Update cumulative time
                time_logs.append(cumulative_time)  # Log the cumulative time
                print(
                    f'It: {it}, Loss: {loss:.3e}, Y0: {Y0_pred:.3f}, Time: {elapsed:.2f}, Learning Rate: {learning_rate:.3e}')
                start_time = time.time()  # Reset the start time for the next print interval

            # Record the average loss and iteration number every 100 iterations
            if it % 500 == 0:
                self.training_loss.append(loss_temp.mean())  # Append the average loss
                loss_temp = np.array([])  # Reset the temporary loss array
                self.iteration.append(it)  # Append the current iteration number

        # Stack the iteration and training loss for plotting
        graph = np.stack((self.iteration, self.training_loss))

        return graph, min_loss, min_loss_state, time_logs

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
        # Abstract method for defining the drift term in the SDE
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Z: Gradient of the function with respect to state variables, size M x D
        # Expected return size: M x 1
        pass

    @abstractmethod
    def g_tf(self, X):  # M x D
        # Abstract method for defining the terminal condition of the SDE
        # Parameter:
        # X: Terminal state variables, size M x D
        # Expected return size: M x 1
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        # Abstract method for defining the drift coefficient of the underlying stochastic process
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Z: Gradient of the function with respect to state variables, size M x D
        # Default implementation returns a zero tensor of size M x D
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        # Abstract method for defining the diffusion coefficient of the underlying stochastic process
        # Parameters:
        # t: Time instances, size M x 1
        # X: State variables, size M x D
        # Y: Function values at state variables, size M x 1
        # Default implementation returns a diagonal matrix of ones of size M x D x D
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)  # M x D x D


class CallOption(FBSNN):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation, correlation_type="no_correlation"):
        super().__init__(Xi, T, M, N, D, Mm, layers, mode, activation, correlation_type)

    # def phi_tf(self, t, X, Y, Z):
    #     # Defines the drift term in the Black-Scholes-Barenblatt equation for a batch
    #     # t: Batch of current times, size M x 1
    #     # X: Batch of current states, size M x D
    #     # Y: Batch of current value functions, size M x 1
    #     # Z: Batch of gradients of the value function with respect to X, size M x D
    #     # Returns the drift term for each instance in the batch, size M x 1
    #     rate = 0.01  # Risk-free interest rate
    #     return rate * (Y)  # M x 1
    # def phi_tf(self, t, X, Y, Z):
    #     r = 0.05
    #     return r * (Y - torch.sum(X * Z, dim=1, keepdim=True))
    def phi_tf(self, t, X, Y, Z):
        r = 0.05
        avg_XZ = torch.sum(X * Z, dim=1, keepdim=True) / self.D
        # return r * (Y - avg_XZ)
        return r * (Y)  # M x 1

    # def g_tf(self, X):
    #     return torch.maximum(torch.sum(X, dim=1, keepdim=True) - self.strike, torch.tensor(0.0).to(self.device))

    def g_tf(self, X):
        avg_X = torch.mean(X, dim=1, keepdim=True)  # Calculate the average of the assets
        return torch.maximum(avg_X - self.strike, torch.tensor(0.0).to(self.device))  # Payoff for the basket option

    def mu_tf(self, t, X, Y, Z):
        # Drift coefficient of the underlying stochastic process for a batch
        # Inherits from the superclass FBSNN without modification
        # Parameters are the same as in phi_tf, with batch sizes
        rate = 0.05
        return rate * X  # M x D

    def sigma_tf(self, t, X, Y):
        # Diffusion coefficient of the underlying stochastic process for a batch
        # t: Batch of current times, size M x 1
        # X: Batch of current states, size M x D
        # Y: Batch of current value functions, size M x 1 (not used in this method)
        # Returns a batch of diagonal matrices, each of size D x D, for the diffusion coefficients
        # Each matrix is scaled by 0.4 times the corresponding state in X
        sigma = 0.20  # Volatility
        return sigma * torch.diag_embed(X)  # M x D x D
    # todo: this sigma definition will need a big change when correlation is introduced


class PredictionGenerator:
    def __init__(self, model, Xi, num_samples):  # todo: check what is this num_samples = 15
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


class BasicOptionPriceCalculator:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma, dimensions, q=0):
        sigma_avg = sigma / np.sqrt(dimensions)
        S_avg = np.mean(S)
        d1 = (np.log(S_avg / K) + (r + 0.5 * sigma_avg ** 2) * T) / (sigma_avg * np.sqrt(T))
        d2 = d1 - sigma_avg * np.sqrt(T)
        call_price = S_avg * normal.cdf(d1) - K * np.exp(-r * T) * normal.cdf(d2)
        delta = normal.cdf(d1)
        return call_price, delta

    def calculate_call_option_prices(self, X_pred, time_array, K, r, sigma, T, dimensions, q=0):
        rows, cols = X_pred.shape

        option_prices = np.zeros((rows, cols))
        deltas = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                if torch.is_tensor(X_pred[i, j]):
                    avg_S = np.mean(X_pred[i, j].detach().numpy())
                else:
                    avg_S = np.mean(X_pred[i, j])

                t = time_array[min(j, len(time_array) - 1)]
                time_to_maturity = T - t
                if time_to_maturity > 0:
                    option_prices[i, j], deltas[i, j] = self.black_scholes_call(avg_S, K, time_to_maturity, r, sigma,
                                                                                dimensions, q)
                else:
                    option_prices[i, j] = max(avg_S - K, 0)
                    if avg_S > K:
                        deltas[i, j] = 1
                    elif avg_S == K:
                        deltas[i, j] = 0.5
                    else:
                        deltas[i, j] = 0

        return option_prices, deltas


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

    def plot_training_loss(self, graph, D, mode, activation, optimizer):
        plt.figure(figsize=self.figsize(1.0))
        plt.plot(graph[0], graph[1])
        plt.xlabel('Iterations')
        plt.ylabel('Value')
        plt.yscale("log")
        plt.title(f'Evolution of the training loss ({mode}-{activation}-{optimizer})')
        filename = f'CallOption{D}DLoss_{mode}_{activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        # plt.close()
        plt.show()

    def plot_prediction(self, t_test, Y_pred, D, model, optimizer):
        samples = min(5, Y_pred.shape[0])
        plt.figure(figsize=self.figsize(1.0))
        for i in range(samples):
            plt.plot(t_test[i, :, 0], Y_pred[i, :], label=f'Sample {i + 1}' if i == 0 else "")
        plt.xlabel('$t$')
        plt.ylabel('$Y_t = u(t,X_t)$')
        plt.title(f'{D}-dimensional Call Option, {model.mode}-{model.activation}-{optimizer}')
        plt.legend()
        filename = f'CallOption{D}D_Prediction_{model.mode}_{model.activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        # plt.close()
        plt.show()

    def plot_exact_vs_learned(self, t_test, Y_pred, Y_test, D, model, optimizer):
        plt.figure(figsize=self.figsize(1.0))
        samples = min(7, Y_pred.shape[0])
        for i in range(samples):
            plt.plot(t_test[i, :, 0], Y_pred[i, :] * 100, 'b', label='Learned $u(t,X_t)$' if i == 0 else "")
            plt.plot(t_test[i, :, 0], Y_test[i, :] * 100, 'r--', label='Exact $u(t,X_t)$' if i == 0 else "")
            plt.plot(t_test[i, -1, 0], Y_test[i, -1] * 100, 'ko', label='$Y_T = u(T,X_T)$' if i == 0 else "")
            plt.plot(t_test[i, 0, 0], Y_pred[i, 0] * 100, 'ks', label='$Y_0 = u(0,X_0)$' if i == 0 else "")

        plt.title(f'{D}-dimensional Basket Option, {model.mode}-{model.activation}-{optimizer}')
        plt.legend()
        plt.xlabel('$t$')
        plt.ylabel('$Y_t = u(t,X_t)$')
        filename = f'CallOption{D}DPreds_{model.mode}_{model.activation}_{optimizer}.png'
        plt.savefig(os.path.join(self.save_path, filename))
        # plt.close()
        plt.show()


class StabilityCheck:
    def __init__(self, model, Xi, perturbation_range, t_test, W_test, save_path, num_points):
        self.model = model
        # Convert Xi to PyTorch tensor if it's not already, and set requires_grad to True
        self.Xi = torch.tensor(Xi, dtype=torch.float32, device=model.device, requires_grad=True) if isinstance(Xi,
                                                                                                               np.ndarray) else Xi.to(
            model.device).requires_grad_(True)
        self.perturbation_range = perturbation_range
        # Convert t_test and W_test to PyTorch tensors if they're not already
        self.t_test = torch.tensor(t_test, dtype=torch.float32, device=model.device) if isinstance(t_test,
                                                                                                   np.ndarray) else t_test.to(
            model.device)
        self.W_test = torch.tensor(W_test, dtype=torch.float32, device=model.device) if isinstance(W_test,
                                                                                                   np.ndarray) else W_test.to(
            model.device)
        self.save_path = save_path
        self.num_points = num_points
        self.num_samples = 16  # This should match the num_samples in PredictionGenerator

    def generate_perturbations(self):
        perturbations = []
        for eps in self.perturbation_range:
            perturbation = self.Xi + eps * torch.randn_like(self.Xi)
            perturbation.requires_grad_(True)  # Ensure the perturbation requires gradients
            perturbations.append(perturbation)
        return perturbations

    def evaluate_perturbations(self, perturbations):
        predictions = []
        for perturbed_Xi in perturbations:
            # Use only the first batch of t_test and W_test
            t_batch = self.t_test[:self.model.M]
            W_batch = self.W_test[:self.model.M]

            # Ensure perturbed_Xi has the correct shape (M, D) and requires gradients
            if perturbed_Xi.shape[0] == 1:
                perturbed_Xi = perturbed_Xi.repeat(self.model.M, 1)
            perturbed_Xi.requires_grad_(True)

            # Run prediction for a single batch
            _, X_pred, Y_pred, _ = self.model.loss_function(t_batch, W_batch, perturbed_Xi)

            # Expand predictions to match the original shape
            X_pred_expanded = X_pred.repeat(self.num_samples, 1, 1)
            Y_pred_expanded = Y_pred.repeat(self.num_samples, 1, 1)

            predictions.append((X_pred_expanded.detach(), Y_pred_expanded.detach()))
        return predictions

    def calculate_relative_errors(self, predictions, Y_test):
        errors = []
        for X_pred, Y_pred in predictions:
            Y_pred_np = Y_pred.cpu().detach().numpy().squeeze()  # Remove extra dimension if exists
            error = np.abs((Y_pred_np - Y_test) / Y_test)
            errors.append(error.mean())
        return errors

    def plot_stability(self, stability_errors_dict, optimizer):
        plt.figure(figsize=(10, 6))
        for key, errors in stability_errors_dict.items():
            mode, activation = key.split("-")
            plt.plot(self.perturbation_range, np.array(errors), marker='o', linestyle='--',
                     label=f'{mode}-{activation}')

        plt.xlabel('Relative distance to the initial condition used for training (%)')
        plt.ylabel('Relative error (%)')
        plt.title(f'{self.model.D}-dimensional Call Option Stability\nOptimizer: {optimizer}')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, f'Stability_{self.model.D}D_{optimizer}.png'))
        # plt.close()
        plt.show()

    def calculate_spectral_radius(self, X):
        # todo: problematic
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.model.device)

        X.requires_grad_(True)

        t = X[:, 0].unsqueeze(1)
        S = X[:, 1:]
        Y, _ = self.model.net_u(t, S)

        # Calculate Jacobian of the system
        jacobian = torch.autograd.functional.jacobian(lambda s: self.model.net_u(t, s)[0], S)

        # Reshape Jacobian
        jacobian_2d = jacobian.reshape(jacobian.shape[0], -1)

        # eigenvalues = torch.linalg.eigvals(jacobian_2d)
        sigular_values = torch.linalg.svd(jacobian_2d).S
        spectral_radius = torch.max(torch.abs(sigular_values))

        return spectral_radius.item()

    def evaluate_stability(self):
        X_samples = torch.rand((self.num_points, self.model.D + 1), device=self.model.device)
        X_samples[:, 0] = torch.rand(self.num_points, device=self.model.device)  # Time component
        X_samples[:, 1:] = 2 * torch.rand((self.num_points, self.model.D),
                                          device=self.model.device) - 1  # State variables

        spec_radius = []
        for X in X_samples:
            spectral_radius = self.calculate_spectral_radius(X.unsqueeze(0))
            spec_radius.append(spectral_radius)
        return spec_radius

    def plot_spectral_radius(self, spectral_radius_dict, optimizer):
        # todo: change the bar plot histogram type of distribution to polar plots
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'pink', 'gray', 'purple', 'brown']

        # Distribution
        plt.figure(figsize=(10, 6))
        for i, (key, spec_radius) in enumerate(spectral_radius_dict.items()):
            mode, activation = key.split("-")
            plt.hist(spec_radius, bins=30, alpha=0.6, label=f'{mode}-{activation}', edgecolor='black',
                     color=colors[i % len(colors)])

        plt.xlabel('Spectral Radius')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Spectral Radii\nOptimizer: {optimizer}')
        plt.axvline(x=1, color='r', linestyle='--', label='Stability Threshold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        plt.savefig(os.path.join(self.save_path, f'SpectralRadius_{optimizer}.png'))
        # plt.close()
        plt.show()

        # Distribution
        plt.figure(figsize=(12, 8))
        for i, (key, spec_radius) in enumerate(spectral_radius_dict.items()):
            mode, activation = key.split("-")
            sns.kdeplot(spec_radius, bw_adjust=1, fill=True, alpha=0.4, label=f'{mode}-{activation}',
                        color=colors[i % len(colors)])

        plt.xlabel('Spectral Radius')
        plt.ylabel('Density (%)')
        plt.title(f'Distribution of Spectral Radii (PDF)\nOptimizer: {optimizer}')
        plt.axvline(x=1, color='r', linestyle='--', label='Stability Threshold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))

        y_ticks = plt.gca().get_yticks()
        plt.gca().set_yticklabels([f'{tick:.1f}%' for tick in y_ticks])

        plt.savefig(os.path.join(self.save_path, f'SpectralRadius_{optimizer}_PDF.png'))
        # plt.close()
        plt.show()

    def cartesian_to_spherical(self, x, y, z):
        x, y, z = np.array(x), np.array(y), np.array(z)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    def plot_spherical_surface(self, spectral_radius_dict, optimizer):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        cmap = plt.get_cmap('plasma')
        colors = cmap(np.linspace(0, 1, len(spectral_radius_dict)))

        for i, (key, spec_radius) in enumerate(spectral_radius_dict.items()):
            x = np.random.uniform(-1, 1, len(spec_radius))
            y = np.random.uniform(-1, 1, len(spec_radius))
            z = np.random.uniform(-1, 1, len(spec_radius))

            r, theta, phi = self.cartesian_to_spherical(x, y, spec_radius)

            x_sph = r * np.sin(theta) * np.cos(phi)
            y_sph = r * np.sin(theta) * np.sin(phi)
            z_sph = r * np.cos(theta)

            surf = ax.plot_trisurf(x_sph, y_sph, z_sph, color=colors[i], alpha=0.6, label=key, linewidth=0.2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Spectral Radius')
        ax.set_title(f'3D Spectral Radius Plot\nOptimizer: {optimizer}')
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Spectral Radius')
        plt.savefig(os.path.join(self.save_path, f'SpectralRadiusSurface_{optimizer}.png'))
        # plt.close()
        plt.show()

    def calculate_jacobian(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32, device=self.model.device)
        X.requires_grad_(True)

        # Split X into time component t and state variables S
        t = X[:, 0].unsqueeze(1)  # Time component (M x 1)
        S = X[:, 1:]  # State variables (M x D)

        # Compute the output of the model and its gradient with respect to S
        Y, _ = self.model.net_u(t, S)

        # Calculate Jacobian only for a subset of points to improve speed
        sample_size = min(100, X.shape[0])
        indices = torch.randperm(X.shape[0])[:sample_size]
        S_sample = S[indices]
        t_sample = t[indices]

        # Compute the Jacobian with respect to the state variables S
        jacobian = torch.autograd.functional.jacobian(lambda s: self.model.net_u(t_sample, s)[0], S_sample)
        return jacobian.cpu().detach().numpy().squeeze()

    def evaluate_jacobian(self):
        # Generate samples for S (state variables) across the D dimensions
        S_samples = torch.linspace(0.5, 1.5, self.num_points, device=self.model.device)

        # Extend S_samples to cover all D dimensions
        S_samples = S_samples.unsqueeze(1).repeat(1, self.model.D)  # Shape: (num_points, D)

        t_samples = torch.linspace(0, self.model.T, self.num_points, device=self.model.device)

        jacobians = []
        for t in tqdm(t_samples, desc="Evaluating Jacobians"):
            # Combine time samples with S_samples
            t_repeated = t.repeat(self.num_points, 1)  # Shape: (num_points, 1)
            X_t = torch.cat((t_repeated, S_samples), dim=1)  # Shape: (num_points, 1 + D)

            jacobian = self.calculate_jacobian(X_t)
            jacobians.append(jacobian)

        return np.array(jacobians).squeeze()

    def plot_jacobian_3d(self, jacobian_dict, optimizer):
        fig = plt.figure(figsize=(20, 15))
        colors = plt.get_cmap('viridis')

        for idx, (key, jacobians) in enumerate(jacobian_dict.items()):
            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')

            if jacobians.ndim == 3:
                jacobians = np.mean(jacobians, axis=2)

            # Apply Gaussian smoothing
            smoothed_jacobians = gaussian_filter(jacobians, sigma=2)

            # Create a higher resolution grid
            x = np.linspace(0.5, 1.5, jacobians.shape[1])
            y = np.linspace(0, self.model.T, jacobians.shape[0])
            X, Y = np.meshgrid(x, y)

            x_new = np.linspace(0.5, 1.5, 100)
            y_new = np.linspace(0, self.model.T, 100)
            X_new, Y_new = np.meshgrid(x_new, y_new)

            # Interpolate the smoothed data onto the higher resolution grid
            Z_new = griddata((X.ravel(), Y.ravel()), smoothed_jacobians.ravel(), (X_new, Y_new), method='cubic')

            # Apply another round of light smoothing
            Z_new = gaussian_filter(Z_new, sigma=0.5)

            # Plot the surface
            surf = ax.plot_surface(X_new, Y_new, Z_new, cmap=colors,
                                   linewidth=0, antialiased=True, alpha=0.8)

            # Add contour lines for better depth perception
            contour = ax.contour(X_new, Y_new, Z_new, zdir='z', offset=Z_new.min(), cmap='coolwarm', alpha=0.5)

            ax.set_xlabel('Underlying Asset Price')
            ax.set_ylabel('Time')
            ax.set_zlabel('Jacobian')
            ax.set_title(f'{key}\nOptimizer: {optimizer}')

            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f'Jacobian3D_Enhanced_{optimizer}.png'), dpi=300)
        plt.show()


class HyperparameterSensitivityAnalyzer:
    def __init__(self, save_path):
        self.save_path = save_path

    def analyze_sensitivity(self, results):
        hyperparams = {k: np.array([r['hyperparams'][k] for r in results]) for k in results[0]['hyperparams']}
        performance_metrics = np.array([r['performance'] for r in results])

        hsic_scores = self.calculate_hsic(hyperparams, performance_metrics)
        self.plot_sensitivity(hyperparams, performance_metrics, hsic_scores)

    def calculate_hsic(self, hyperparams, performance_metrics):
        return {param: self._calculate_hsic_single(values, performance_metrics)
                for param, values in hyperparams.items()}

    def _calculate_hsic_single(self, X, Y):
        n = X.shape[0]
        X = X.reshape(-1, 1)
        Y = Y.reshape(-1, 1)

        dist_X = squareform(pdist(X, metric='euclidean'))
        dist_Y = squareform(pdist(Y, metric='euclidean'))

        K_X = np.exp(-dist_X ** 2 / (2 * np.median(dist_X) ** 2))
        K_Y = np.exp(-dist_Y ** 2 / (2 * np.median(dist_Y) ** 2))

        H = np.eye(n) - np.ones((n, n)) / n
        K_X_centered = H @ K_X @ H
        K_Y_centered = H @ K_Y @ H

        return np.sum(K_X_centered * K_Y_centered) / (n ** 2)

    def plot_sensitivity(self, hyperparams, performance_metrics, hsic_scores):
        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        axes = axes.flatten()

        # HSIC Scores
        ax = axes[0]
        ax.barh(list(hsic_scores.keys()), list(hsic_scores.values()))
        ax.set_title('HSIC Scores')
        ax.set_xlabel('HSIC Score')

        # Histograms
        for i, (param, values) in enumerate(hyperparams.items()):
            if i + 1 >= len(axes):
                break
            ax = axes[i + 1]
            top_mask = performance_metrics <= np.percentile(performance_metrics, 10)
            bottom_mask = performance_metrics >= np.percentile(performance_metrics, 90)

            sns.histplot(values[top_mask], kde=True, color='blue', label='Top 10% Performers', ax=ax, alpha=0.6)
            sns.histplot(values[bottom_mask], kde=True, color='red', label='Bottom 10% Performers', ax=ax, alpha=0.6)

            ax.set_title(f'Distribution of {param}')
            ax.set_xlabel(param)
            ax.set_ylabel('Density')
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'{self.save_path}/sensitivity_analysis.png')
        plt.show()
        # plt.close()


class SensitivityExecutor:
    def __init__(self, Xi, T, N, D, M, layers_template, mode, activation, optimizer, save_path):
        self.Xi = Xi
        self.T = T
        self.N = N
        self.D = D
        self.M = M
        self.layers_template = layers_template
        self.mode = mode
        self.activation = activation
        self.optimizer = optimizer
        self.save_path = save_path
        self.sensitivity_analyzer = HyperparameterSensitivityAnalyzer(save_path)

    def execute(self, num_real_samples=5, num_simulated_samples=995, num_simulation_runs=20):
        real_results = self.run_real_samples(num_real_samples)
        averaged_simulated_results = self.run_multiple_simulations(num_simulated_samples, num_simulation_runs)
        all_results = real_results + averaged_simulated_results
        self.sensitivity_analyzer.analyze_sensitivity(all_results)

    def run_real_samples(self, num_samples):
        hyperparams_list = [self.sample_hyperparameters() for _ in range(num_samples)]
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.evaluate_model, hyperparams) for hyperparams in hyperparams_list]

            results = []
            for future in as_completed(futures):
                try:
                    hyperparams, performance, _ = future.result()
                    results.append({
                        'hyperparams': hyperparams,
                        'performance': performance
                    })
                    print(f"Completed real sample with performance: {performance}")
                except Exception as e:
                    print(f"An error occurred in real sample: {e}")

        return results

    def run_multiple_simulations(self, num_samples, num_runs):
        all_simulated_results = []
        for _ in range(num_runs):
            all_simulated_results.extend(self.run_simulated_samples(num_samples))

        # Group results by hyperparameters
        grouped_results = {}
        for result in all_simulated_results:
            key = tuple(result['hyperparams'].items())
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result['performance'])

        # Average the performances
        averaged_results = []
        for hyperparam_tuple, performances in grouped_results.items():
            averaged_results.append({
                'hyperparams': dict(hyperparam_tuple),
                'performance': np.mean(performances)
            })

        return averaged_results

    def run_simulated_samples(self, num_samples):
        hyperparams = self.mock_sample_hyperparameters(num_samples)
        performance = self.simulate_performance(hyperparams)
        return [{'hyperparams': {k: v[i] for k, v in hyperparams.items()},
                 'performance': performance[i]}
                for i in range(num_samples)]

    def sample_hyperparameters(self):
        return {
            'initial_lr': loguniform.rvs(1e-6, 1e-2),
            'fine_tuning_lr': loguniform.rvs(1e-7, 1e-3),
            'initial_n_iter': np.random.randint(5000, 30000),
            'fine_tuning_n_iter': np.random.randint(1000, 6000)
        }

    def mock_sample_hyperparameters(self, num_samples):
        initial_lr = np.random.uniform(1e-6, 1e-2, num_samples)
        return {
            'initial_lr': initial_lr,
            'fine_tuning_lr': initial_lr * np.random.uniform(0.1, 0.5, num_samples),
            'initial_n_iter': np.random.randint(20000, 100000, num_samples),
            'fine_tuning_n_iter': np.random.randint(5000, 20000, num_samples)
        }

    def simulate_performance(self, hyperparams):
        performance = (
                -20 * (np.log10(hyperparams['initial_lr']) + 4) ** 2 +
                -10 * (np.log10(hyperparams['fine_tuning_lr']) + 5) ** 2 +
                np.log(hyperparams['initial_n_iter']) * 2 +
                np.log(hyperparams['fine_tuning_n_iter'])
        )
        performance += np.random.normal(0, 2, performance.shape)
        return -performance  # Lower is better

    def evaluate_model(self, hyperparams):
        layers = [self.D + 1] + self.layers_template + [1]
        Xi = np.array([1.0] * self.D)[None, :]
        model = CallOption(Xi, self.T, self.M, self.N, self.D, int(self.N ** (1/5)), layers, self.mode, self.activation)

        trainer = TrainingPhases(model)

        graph_initial = trainer.train_initial_phase(
            hyperparams['initial_n_iter'], hyperparams['initial_lr'], self.optimizer)

        graph_finetune = trainer.fine_tuning_phase(
            hyperparams['fine_tuning_n_iter'], hyperparams['fine_tuning_lr'], self.optimizer)

        # Handle the case where we might have scalar values instead of arrays
        if np.isscalar(graph_initial[1]) and np.isscalar(graph_finetune[1]):
            combined_graph = np.array([graph_initial[1], graph_finetune[1]])
        else:
            combined_graph = np.concatenate((graph_initial[1], graph_finetune[1]))

        min_loss = np.min(combined_graph)

        return hyperparams, min_loss, combined_graph


class FastSensitivityExecutor:
    def __init__(self, save_path):
        self.save_path = save_path

    def execute(self, num_samples=10000):
        hyperparams = self.sample_hyperparameters(num_samples)
        performance = self.simulate_performance(hyperparams)
        hsic_scores = self.calculate_hsic(hyperparams, performance)
        permutation_importances = self.calculate_permutation_importance(hyperparams, performance)
        self.plot_sensitivity(hyperparams, performance, hsic_scores, permutation_importances)

    def sample_hyperparameters(self, num_samples):
        return {
            # Sample learning rates from a log-uniform distribution
            'initial_lr': loguniform.rvs(1e-4, 1e-2, size=num_samples),
            'fine_tuning_lr': loguniform.rvs(1e-6, 1e-4, size=num_samples),
            'initial_n_iter': np.random.randint(20000, 50000, num_samples),
            'fine_tuning_n_iter': np.random.randint(5000, 15000, num_samples)
        }

    def simulate_performance(self, hyperparams):
        performance = (
            -np.log(hyperparams['initial_lr']) * 0.1 +
            -np.log(hyperparams['fine_tuning_lr']) * 0.05 +
            np.log(hyperparams['initial_n_iter']) * 0.2 +
            np.log(hyperparams['fine_tuning_n_iter']) * 0.1
        )
        performance += np.random.normal(0, 0.5, len(performance))
        return performance

    def calculate_hsic(self, hyperparams, performance):
        def rank_correlation(x, y):
            return np.abs(np.corrcoef(np.argsort(x), np.argsort(y))[0, 1])
        return {param: rank_correlation(values, performance) for param, values in hyperparams.items()}

    def calculate_permutation_importance(self, hyperparams, performance):
        baseline_performance = np.mean(performance)
        permutation_importances = {}

        for param, values in hyperparams.items():
            permuted_performance = np.zeros_like(performance)

            for i in range(len(values)):
                permuted_values = values.copy()
                np.random.shuffle(permuted_values)
                permuted_performance[i] = self.simulate_performance({
                    'initial_lr': hyperparams['initial_lr'] if param != 'initial_lr' else permuted_values,
                    'fine_tuning_lr': hyperparams['fine_tuning_lr'] if param != 'fine_tuning_lr' else permuted_values,
                    'initial_n_iter': hyperparams['initial_n_iter'] if param != 'initial_n_iter' else permuted_values,
                    'fine_tuning_n_iter': hyperparams['fine_tuning_n_iter'] if param != 'fine_tuning_n_iter' else permuted_values
                })[i]

            permutation_importances[param] = baseline_performance - np.mean(permuted_performance)

        return permutation_importances

    def plot_sensitivity(self, hyperparams, performance, hsic_scores, permutation_importances):
        # Set up a grid with a large left plot and a 2x2 grid of smaller right plots
        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(2, 3, width_ratios=[1, 2, 2])

        # Sensitivity Scores on the left (spanning both rows)
        ax1 = fig.add_subplot(gs[:, 0])
        ax1.barh(list(hsic_scores.keys()), list(hsic_scores.values()), color='skyblue', label='HSIC Score')
        ax1.barh(list(permutation_importances.keys()), list(permutation_importances.values()), color='lightcoral',
                 alpha=0.6, label='Permutation Importance')
        ax1.set_title('Sensitivity Scores')
        ax1.set_xlabel('Score')
        ax1.legend()

        # Create the 2x2 grid on the right for the distributions
        for i, (param, values) in enumerate(hyperparams.items()):
            ax = fig.add_subplot(gs[i // 2, (i % 2) + 1])  # Adjust index to place in 2x2 grid
            top_mask = performance <= np.percentile(performance, 10)
            bottom_mask = performance >= np.percentile(performance, 90)

            sns.histplot(values[top_mask], kde=True, color='blue', label='Top 10% Performers', ax=ax, alpha=0.6,
                         bins=30, stat='percent')
            sns.histplot(values[bottom_mask], kde=True, color='red', label='Bottom 10% Performers', ax=ax, alpha=0.6,
                         bins=30, stat='percent')

            ax.set_title(f'Distribution of {param}', fontsize=12, pad=20)
            ax.set_xlabel(param, fontsize=11)
            ax.set_ylabel('Density (%)', fontsize=11)
            ax.legend()

            # Ensure y-axis starts at 0
            ax.set_ylim(bottom=0)

        plt.tight_layout(h_pad=5.0, w_pad=3.0)
        plt.savefig(f'{self.save_path}/sensitivity_analysis_new.png')
        plt.show()


class TrainingExecutor:
    def __init__(self, Xi, T, Ms, N, Ds, layers_template, modes, activations, optimizers, save_path, perturbation_range,
                 num_points):
        self.Xi = Xi
        self.T = T
        self.Ms = Ms
        self.N = N
        self.Ds = Ds
        self.layers_template = layers_template
        self.modes = modes
        self.activations = activations
        self.optimizers = optimizers
        self.save_path = save_path
        self.perturbation_range = perturbation_range
        self.num_points = num_points
        # self.initial_lrs = initial_lrs
        # self.fine_tuning_lrs = fine_tuning_lrs
        # self.initial_iters = initial_iters
        # self.fine_tuning_iters = fine_tuning_iters
        self.results_df = pd.DataFrame(columns=[
            "Batch Size (M)", "Dimension (D)", "Initial Learning Rate", "Fine-Tuning Learning Rate",
            "Initial Iterations", "Fine-Tuning Iterations",
            "Combination Type", "Optimizer", "Mean Error", "Std Error", "RMSE", "Min Loss", "Exact Option Price",
            "Learned Option Price",
            "Total Run Time", "Spectral Radius"])

    def execute(self):
        # todo: Software enginerring work to better code this big loop (multi-threading, see draft code)
        for M in self.Ms:
            for D in self.Ds:
                for optimizer in self.optimizers:

                    stability_errors_dict = {}
                    spectral_radius_dict = {}
                    jacobian_dict = {}
                    for mode in tqdm(self.modes, desc="Network Modes", leave=False):
                        for activation in tqdm(self.activations, desc="Activation Functions", leave=False):
                            start_time = time.time()
                            print(
                                f"Running combination: Mode={mode}, Activation={activation}, Optimizer={optimizer}, M={M}, D={D}")

                            # Define layers and Xi based on D
                            layers = [D + 1] + self.layers_template + [1]
                            Xi = np.array([1.0] * D)[None, :]

                            initial_lr = 1e-3  # Example value; replace with optimal value after sensitivity analysis
                            fine_tuning_lr = 1e-4  # Example value; replace with optimal value after sensitivity analysis
                            initial_n_iter = 3000  # Example value; replace with optimal value after sensitivity analysis
                            fine_tuning_n_iter = 1000  # Example value; replace with optimal value after sensitivity analysis

                            model = CallOption(Xi, self.T, M, initial_n_iter, D, int(self.N ** (1 / 5)), layers, mode,
                                               activation)
                            trainer = TrainingPhases(model)

                            # Initial training phase
                            graph_initial = trainer.train_initial_phase(initial_n_iter, initial_lr, optimizer)

                            # Fine-tuning training phase
                            graph_finetune = trainer.fine_tuning_phase(fine_tuning_n_iter, fine_tuning_lr, optimizer)

                            combine_graph = np.concatenate((graph_initial[1], graph_finetune[1]))

                            min_loss = np.min(combine_graph)
                            print(f"Minimum Loss: {min_loss}")

                            predictor = PredictionGenerator(model, Xi, num_samples=16)
                            t_test, W_test, X_pred, Y_pred = predictor.generate_predictions()

                            price_calculator = BasicOptionPriceCalculator()
                            # price_calculator = BasketOptionPriceCalculator()
                            K = 1.0
                            r = 0.05
                            sigma = 0.20
                            q = 0
                            T = 1.0

                            Y_test, Z_test = price_calculator.calculate_call_option_prices(X_pred[:, :, 0],
                                                                                           t_test[:, 0, 0],
                                                                                           K, r, sigma, T, D)

                            Y_test = np.array(Y_test)

                            if Y_pred.ndim == 3:
                                Y_pred = Y_pred[:, :, 0]

                            # Calculate Errors
                            errors = (Y_test.squeeze() - Y_pred) ** 2
                            mean_error = errors.mean()
                            std_error = errors.std()
                            rmse = np.sqrt(errors.mean())

                            plotter = TrainingPlot(self.save_path)
                            plotter.plot_training_loss((model.iteration, model.training_loss), D, mode, activation,
                                                       optimizer)
                            plotter.plot_exact_vs_learned(t_test, Y_pred, Y_test, D, model, optimizer)

                            Xi_tensor = torch.tensor(Xi, dtype=torch.float32, device=model.device,
                                                     requires_grad=True) if isinstance(Xi, np.ndarray) else Xi.to(
                                model.device).requires_grad(True)

                            stability_checker = StabilityCheck(model, Xi_tensor, self.perturbation_range, t_test,
                                                               W_test, self.save_path, self.num_points)
                            perturbed_Xi = stability_checker.generate_perturbations()
                            perturbed_predictions = stability_checker.evaluate_perturbations(perturbed_Xi)
                            stability_errors = stability_checker.calculate_relative_errors(perturbed_predictions,
                                                                                           Y_test)

                            combination_key = f"{mode}-{activation}"
                            stability_errors_dict[combination_key] = stability_errors

                            spec_radius = stability_checker.evaluate_stability()
                            spectral_radius_dict[combination_key] = spec_radius

                            jacobians = stability_checker.evaluate_jacobian()
                            # Reduce dimensionality if D > 1
                            if jacobians.ndim == 4:  # Check if the Jacobians have an extra dimension
                                jacobians = np.linalg.norm(jacobians, axis=-1)
                            # Take the norm across the last dimension

                            jacobian_dict[combination_key] = jacobians

                            mean_radius = np.mean(spec_radius)

                            learned_option_price = max(0.0, Y_pred[0, 0])
                            exact_option_price = Y_test[
                                0, 0]  # todo: this is the real option price we need to do for L2 error calculations
                            total_run_time = time.time() - start_time
                            combination_type = f"{mode}_{activation}"

                            new_row = {
                                "Batch Size (M)": M,
                                "Dimension (D)": D,
                                "Initial Learning Rate": initial_lr,
                                "Fine-Tuning Learning Rate": fine_tuning_lr,
                                "Initial Iterations": initial_n_iter,
                                "Fine-Tuning Iterations": fine_tuning_n_iter,
                                "Combination Type": combination_type,
                                "Optimizer": optimizer,
                                "Mean Error": mean_error,
                                "Std Error": std_error,
                                "RMSE": rmse,
                                "Min Loss": min_loss,
                                "Exact Option Price": exact_option_price,
                                "Learned Option Price": learned_option_price,
                                "Total Run Time": total_run_time,
                                "Spectral Radius": mean_radius
                            }
                            self.results_df = pd.concat([self.results_df, pd.DataFrame([new_row])],
                                                        ignore_index=True)
                    stability_checker.plot_stability(stability_errors_dict, optimizer)
                    stability_checker.plot_spectral_radius(spectral_radius_dict, optimizer)
                    stability_checker.plot_spherical_surface(spectral_radius_dict, optimizer)
                    stability_checker.plot_jacobian_3d(jacobian_dict, optimizer)

                self.results_df.to_csv(os.path.join(self.save_path, 'results.csv'), index=False)
                print("Results saved to results.csv")
                print(self.results_df)


if __name__ == "__main__":
    # Parameters
    N = 50
    D = 5
    M = 32
    layers_template = 4 * [256]
    Xi = np.array([1.0] * D)[None, :]
    T = 1.0
    mode = "Naisnet"
    activation = "Sine"
    optimizer = "Adam"

    save_path = r'C:/Users/aa04947/OneDrive - APG/Desktop/dnnpde_output'
    total_start_time = time.time()

    # Create SensitivityExecutor instance
    # sensitivity_executor = SensitivityExecutor(Xi, T, N, D, M, layers_template, mode, activation, optimizer, save_path)
    # Create FastSensitivityExecutor instance
    sensitivity_executor = FastSensitivityExecutor(save_path)

    # Run sensitivity analysis
    print("Starting sensitivity analysis...")
    sensitivity_executor.execute(num_samples=10000)
    print("Sensitivity analysis completed.")


    # Run sensitivity analysis
    # print("Starting sensitivity analysis...")
    # sensitivity_executor.execute(num_real_samples = 0, num_simulated_samples=1000, num_simulation_runs=10)  # Increased number of samples
    # print("Sensitivity analysis completed.")

    total_run_time = time.time() - total_start_time
    print(f"Total run time for the entire algorithm: {total_run_time:.2f} seconds")


# if __name__ == "__main__":
#     # Ms = [1, 2, 5, 8, 16, 32, 64, 100, 128, 1000, 10000]  # Different batch sizes / # MC
#     Ms = [32, 128]
#     N = 50
#     # Ds = [1, 3, 5, 10, 20, 50, 100]  # Different dimensions
#     Ds = [1, 5]
#     layers_template = 4 * [256]  # Example layers template
#     Xi = np.array([1.0] * max(Ds))[None, :]  # Initial Xi, will be adjusted dynamically
#     T = 1.0
#     perturbation_range = np.linspace(0, 0.8, 10)
#
#     modes = ["Naisnet"]
#     activations = ["Sine"]
#     optimizers = ["Adam"]
#
#     save_path = r'C:/Users/aa04947/OneDrive - APG/Desktop/dnnpde_output/s1'
#
#     # initial_lrs = [1e-3]  # Different initial learning rates
#     # fine_tuning_lrs = [1e-5]  # Different fine-tuning learning rates
#     # initial_iters = [2000]  # Different initial iterations
#     # fine_tuning_iters = [500]  # Different fine-tuning iterations
#
#     # initial_lrs = [1e-3, 5e-3, 1e-2, 2e-3, 5e-4, 1e-1, 1e-3, 7e-3, 3e-3, 8e-3]  # Different initial learning rates
#     # fine_tuning_lrs = [1e-5, 5e-5, 1e-4, 2e-5, 5e-6, 1e-3, 1e-5, 7e-5, 3e-5, 8e-5]  # Different fine-tuning learning rates
#     # initial_iters = [1000, 3000, 5000, 7000, 9000, 12000, 15000, 18000, 21000, 25000]  # Different initial iterations
#     # fine_tuning_iters = [200, 600, 1000, 1400, 1800, 2400, 3000, 3600, 4200, 5000]  # Different fine-tuning iterations
#
#     total_start_time = time.time()
#
#     # Create SensitivityExecutor instance
#     sensitivity_executor = SensitivityExecutor(Xi, T, Ms, N, Ds, layers_template, modes, activations, optimizers,
#                                                save_path)
#
#     # Run sensitivity analysis
#     print("Starting sensitivity analysis...")
#     sensitivity_executor.execute(num_samples=20)  # Adjust number of samples as needed
#     print("Sensitivity analysis completed.")
#
#     # Run sensitivity analysis using the new SensitivityExecutor
#     # training_executor = TrainingExecutor(Xi, T, Ms, N, Ds, layers_template, modes, activations, optimizers, save_path, perturbation_range, num_points=20, initial_lrs=initial_lrs, fine_tuning_lrs=fine_tuning_lrs, initial_iters=initial_iters, fine_tuning_iters=fine_tuning_iters)
#
#     # Step 1: Run sensitivity analysis using SensitivityExecutor
#     training_executor = TrainingExecutor(Xi, T, Ms, N, Ds, layers_template, modes, activations, optimizers, save_path,
#                                          perturbation_range, num_points=20)
#
#     # Step 2: After analyzing the sensitivity results, manually set the best hyperparameters
#     # optimal_initial_lr = 1e-3  # Replace with the best value found
#     # optimal_fine_tuning_lr = 1e-4  # Replace with the best value found
#     # optimal_initial_n_iter = 3000  # Replace with the best value found
#     # optimal_fine_tuning_n_iter = 1000  # Replace with the best value found
#     #
#     # # Step 3: Run TrainingExecutor with the selected optimal hyperparameters
#     # training_executor.train(optimal_initial_n_iter + optimal_fine_tuning_n_iter, optimal_initial_lr, 'Adam')
#
#     total_run_time = time.time() - total_start_time
#     print(f"Total run time for the entire algorithm: {total_run_time:.2f} seconds")




