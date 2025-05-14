import os
import sys
import time
import tqdm
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy.stats import multivariate_normal as normal
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
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
        self.strike = 1.0 * self.D  # strike price

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

        # todo: add the Resnet method or other NN architectures

        # Apply a custom weights initialization to the model.
        self.model.apply(self.weights_init)

        # Initialize lists to record training loss and iterations.
        self.training_loss = []
        self.iteration = []
        self.correlation_matrix = torch.eye(self.D).to(self.device)  # Placeholder for correlation matrix

    def weights_init(self, m):
        # Custom weight initialization method for neural network layers
        # Parameters:
        # m: A layer of the neural network

        if type(m) == nn.Linear:
            # Initialize the weights of the linear layer using Xavier uniform initialization
            torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):  # M x 1, M x D
        # Debug print
        print(f"Initial shapes - t: {t.shape}, X: {X.shape}")

        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if X.dim() == 1:
            X = X.unsqueeze(-1)

        # Debug print
        print(f"Before cat - t: {t.shape}, X: {X.shape}")

        # Concatenate the time and state variables along second dimension
        input = torch.cat((t, X), 1)

        # Debug print
        print(f"After cat - input: {input.shape}")

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
        DW[:, 1:,
        :] = DW_uncorrelated  # todo: np.einsum('ij,mnj->mni', self.L, DW_uncorrelated) # Apply Cholesky matrix to introduce correlations

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
            if it % 100 == 0:
                elapsed = time.time() - start_time  # Calculate the elapsed time
                print(
                    f'It: {it}, Loss: {loss:.3e}, Y0: {Y0_pred:.3f}, Time: {elapsed:.2f}, Learning Rate: {learning_rate:.3e}')
                start_time = time.time()  # Reset the start time for the next print interval

            # Record the average loss and iteration number every 100 iterations
            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())  # Append the average loss
                loss_temp = np.array([])  # Reset the temporary loss array
                self.iteration.append(it)  # Append the current iteration number

        # Stack the iteration and training loss for plotting
        graph = np.stack((self.iteration, self.training_loss))

        return graph, min_loss, min_loss_state

    def predict(self, Xi_star, t_star, W_star):
        print(f"Predict input shapes - Xi_star: {Xi_star.shape}, t_star: {t_star.shape}, W_star: {W_star.shape}")

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

        print(f"Before loss_function - Xi_star: {Xi_star.shape}, t_star: {t_star.shape}, W_star: {W_star.shape}")

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
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, Mm, layers, mode, activation)
        self.M = M  # Ensure batch size is initialized

    # def phi_tf(self, t, X, Y, Z):
    #     # Defines the drift term in the Black-Scholes-Barenblatt equation for a batch
    #     # t: Batch of current times, size M x 1
    #     # X: Batch of current states, size M x D
    #     # Y: Batch of current value functions, size M x 1
    #     # Z: Batch of gradients of the value function with respect to X, size M x D
    #     # Returns the drift term for each instance in the batch, size M x 1
    #     rate = 0.01  # Risk-free interest rate
    #     return rate * (Y)  # M x 1
    def phi_tf(self, t, X, Y, Z):
        r = 0.05
        return r * (Y - torch.sum(X * Z, dim=1, keepdim=True))

    def g_tf(self, X):
        return torch.maximum(torch.sum(X, dim=1, keepdim=True) - self.strike, torch.tensor(0.0).to(self.device))

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
    #todo: this sigma definition will need a big change when correlation is introduced


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


class VanillaOptionPriceCalculator:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma, q=0):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = (S * np.exp(-q * T) * normal.cdf(d1)) - (K * np.exp(-r * T) * normal.cdf(d2))
        delta = normal.cdf(d1)
        return call_price, delta

    def calculate_option_prices(self, X_pred, time_array, K, r, sigma, T, q=0):
        rows, cols = X_pred.shape  # Extract rows and cols correctly
        option_prices = np.zeros((rows, cols))
        deltas = np.zeros((rows, cols))

        for i in range(rows):
            for j in range(cols):
                S = X_pred[i, j].detach().numpy() if torch.is_tensor(X_pred[i, j]) else X_pred[i, j]
                # t = time_array[j]  # Ensure correct indexing
                t = time_array[min(j, len(time_array) - 1)]  # Ensure correct indexing
                time_to_maturity = T - t
                if time_to_maturity > 0:
                    option_prices[i, j], deltas[i, j] = self.black_scholes_call(S, K, time_to_maturity, r, sigma, q)
                else:
                    option_prices[i, j] = max(S - K, 0)
                    if S > K:
                        deltas[i, j] = 1
                    elif S == K:
                        deltas[i, j] = 0.5
                    else:
                        deltas[i, j] = 0

        return option_prices, deltas


class BasketOptionPriceCalculator:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma, q=0):
        # Ensure all inputs are tensors
        S, K, T, r, sigma, q = [torch.as_tensor(x).to(S.device) for x in [S, K, T, r, sigma, q]]

        d1 = (torch.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * torch.sqrt(T))
        d2 = d1 - sigma * torch.sqrt(T)

        normal = torch.distributions.Normal(0, 1)
        call_price = S * torch.exp(-q * T) * normal.cdf(d1) - K * torch.exp(-r * T) * normal.cdf(d2)
        delta = normal.cdf(d1)

        return call_price, delta

    @staticmethod
    def calculate_option_prices(S, t, K, r, sigma, T):
        # S shape: [batch_size, num_time_steps, num_assets]
        # t shape: [batch_size, num_time_steps, 1]
        batch_size, num_time_steps, num_assets = S.shape

        time_to_maturity = T - t

        # Reshape S and time_to_maturity for vectorized calculation
        S_flat = S.reshape(-1, num_assets)  # [batch_size * num_time_steps, num_assets]
        T_flat = time_to_maturity.reshape(-1, 1).expand(-1, num_assets)  # [batch_size * num_time_steps, num_assets]

        option_prices, deltas = BasketOptionPriceCalculator.black_scholes_call(S_flat, K, T_flat, r, sigma)

        # Reshape results back to original dimensions
        option_prices = option_prices.reshape(batch_size, num_time_steps, num_assets)
        deltas = deltas.reshape(batch_size, num_time_steps, num_assets)

        # Calculate basket option price and delta (equal-weighted average)
        basket_price = torch.mean(option_prices, dim=-1, keepdim=True)
        basket_delta = torch.mean(deltas, dim=-1, keepdim=True)

        return basket_price, basket_delta


class TrainingPhases:
    def __init__(self, model):
        self.model = model

    def train_initial_phase(self, n_iter, lr, optimizer_type='Adam'):
        print("Starting initial training phase...")
        tot = time.time()
        print(self.model.device)
        graph, min_loss, min_loss_state = self.model.train(n_iter, lr, optimizer_type)
        print("Initial training phase completed. Total time:", time.time() - tot, "s")
        self.min_loss = min_loss
        self.min_loss_state = min_loss_state
        return graph

    def fine_tuning_phase(self, n_iter, lr, optimizer_type='Adam'):
        print("Starting fine-tuning phase...")
        tot = time.time()
        print(self.model.device)
        graph, min_loss, min_loss_state = self.model.train(n_iter, lr, optimizer_type)
        print("Fine-tuning phase completed. Total time:", time.time() - tot, "s")
        self.min_loss = min_loss
        self.min_loss_state = min_loss_state
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
            plt.plot(t_test[i, :, 0], Y_pred[i, :], label=f'Sample {i+1}' if i == 0 else "")
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



if __name__ == "__main__":
    # Hyperparameters
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions
    Mm = N ** (1 / 5)
    layers = [D + 1] + 4 * [256] + [1]
    Xi = np.array([1.0] * D)[None, :]
    T = 1.0
    perturbation_range = np.linspace(0, 0.8, 10)  # Example perturbation range from 0% to 50%

    # Available architectures, activations, and optimizers
    modes = ["Naisnet", "FC"]
    activations = ["Sine", "ReLU"]
    optimizers = ["Adam"]

    # Initialize DataFrame to record metrics
    columns = ["Batch Size (M)", "Dimension", "Combination Type", "Optimizer", "Mean Error", "Std Error", "RMSE", "Min Loss",
               "Exact Option Price", "Learned Option Price", "Total Run Time", "Spectral Radius"]
    results_df = pd.DataFrame(columns=columns)

    # Path to save output plots
    save_path = r'C:/Users/aa04947/OneDrive - APG/Desktop/dnnpde_output/archive'

    # Loop through all combinations
    for optimizer in optimizers:
        # Define empty dictionaries to store results
        stability_errors_dict = {}
        spectral_radius_dict = {}
        jacobian_dict = {}

        for mode in tqdm(modes, desc = "Network Modes", leave = False):
            for activation in tqdm(activations, desc = "Activation Functions", leave = False):
                start_time = time.time()
                print(f"Running combination: Mode={mode}, Activation={activation}, Optimizer={optimizer}")

                # Initialize model
                model = CallOption(Xi, T, M, N, D, Mm, layers, mode, activation)

                # Create an instance of TrainingPlot
                plotter = TrainingPlot(save_path)

                # Create an instance of TrainingPhases
                trainer = TrainingPhases(model)

                # Initial training phase
                initial_n_iter = 2000
                initial_lr = 1e-3
                trainer.train_initial_phase(initial_n_iter, initial_lr)

                # Fine-tuning phase
                fine_tuning_n_iter = 500
                fine_tuning_lr = 1e-5
                trainer.fine_tuning_phase(fine_tuning_n_iter, fine_tuning_lr)

                min_loss = trainer.min_loss
                print(f"Minimum Loss: {min_loss}")

                # Generate predictions using PredictionGenerator
                predictor = PredictionGenerator(model, Xi, num_samples=16)
                t_test, W_test, X_pred, Y_pred = predictor.generate_predictions()

                # Calculate option prices
                # price_calculator = VanillaOptionPriceCalculator()
                price_calculator = BasketOptionPriceCalculator()
                K = 1.0  # Strike price
                r = 0.05  # Risk-free interest rate
                sigma = 0.20  # Volatility
                q = 0  # Dividend yield (assuming none)
                T = 1  # Expiry time in years
                num_points = 20  # For stability check and spectral radius calculations

                # Vanilla Option case
                # Y_test, Z_test = price_calculator.calculate_option_prices(X_pred[:, :, 0], t_test[:, 0, 0], K, r, sigma,
                #                                                           T, q)
                # Ensure Y_test is a NumPy array
                # Y_test = np.array(Y_test)
                #
                # # Calculate errors
                # errors = (Y_test[:500] - Y_pred[:500, :, 0]) ** 2
                # mean_error = errors.mean()
                # std_error = errors.std()
                # rmse = np.sqrt(errors.mean())

                # Basket Option case
                # Convert predictions to tensors -> Only for basket option cases
                X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).to(model.device)
                t_test_tensor = torch.tensor(t_test, dtype=torch.float32).to(model.device)

                print(f"X_pred_tensor shape: {X_pred_tensor.shape}")
                print(f"t_test_tensor shape: {t_test_tensor.shape}")

                Y_test, Z_test = price_calculator.calculate_option_prices(X_pred_tensor, t_test_tensor, K, r, sigma, T)

                print(f"Y_test shape: {Y_test.shape}")
                print(f"Z_test shape: {Z_test.shape}")

                # Ensure Y_test is a numpy array
                Y_test = Y_test.cpu().numpy()
                Z_test = Z_test.cpu().numpy()

                # Ensure Y_pred has the correct shape for comparison
                if Y_pred.ndim == 3:
                    Y_pred = Y_pred[:, :, 0] # This reshapes Y_pred to (M * num_sample, 4) for example

                # Calculate errors
                errors = (Y_test.squeeze() - Y_pred) ** 2 # nparray (M * num_sample, 4)
                x = Y_test.squeeze()
                mean_error = errors.mean()
                std_error = errors.std()
                rmse = np.sqrt(errors.mean())

                print(f"Mean Error: {mean_error}")
                print(f"Std Error: {std_error}")
                print(f"RMSE: {rmse}")

                # Use TrainingPlot instance to generate and save plots
                plotter = TrainingPlot(save_path)
                plotter.plot_training_loss((model.iteration, model.training_loss), D, mode, activation, optimizer)
                plotter.plot_prediction(t_test, Y_pred, D, model, optimizer)
                plotter.plot_exact_vs_learned(t_test, Y_pred, Y_test[:, :, 0], D, model, optimizer)

                # Store results for plotting
                combination_key = f"{mode}-{activation}"
                # Ensure learned option price is non-negative
                learned_option_price = max(0.0, Y_pred[0, 0])  # Extract learned option price at t=0
                exact_option_price = Y_test[0, 0, 0]
                total_run_time = time.time() - start_time
                combination_type = f"{mode}_{activation}"

                new_row = {
                    "Batch Size (M)": M,
                    "Dimension": D,
                    "Combination Type": combination_type,
                    "Optimizer": optimizer,
                    "Mean Error": mean_error,
                    "Std Error": std_error,
                    "RMSE": rmse,
                    "Min Loss": min_loss,
                    "Exact Option Price": exact_option_price,
                    "Learned Option Price": learned_option_price,
                    "Total Run Time": total_run_time
                }
                results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)

    # Save results to a CSV file
    results_df.to_csv(os.path.join(save_path, 'results_nbs.csv'), index=False)
    print("Results saved to results.csv")
    print(results_df)







