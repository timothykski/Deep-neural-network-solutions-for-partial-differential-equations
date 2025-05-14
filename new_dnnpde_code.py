import os
import numpy as np
from abc import ABC, abstractmethod
import time
from scipy.stats import multivariate_normal as normal
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)


class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module"""
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)

class Resnet(nn.Module):
    def __init__(self, layers, activation):
        super(Resnet, self).__init__()
        self.activation_function = activation

        self.input_layer = nn.Linear(layers[0], layers[1])
        self.hidden_layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(1, len(layers) - 2)])
        self.output_layer = nn.Linear(layers[-2], layers[-1])

    def forward(self, x):
        out = self.input_layer(x)
        out = self.activation_function(out)

        for layer in self.hidden_layers:
            shortcut = out.clone()
            out = layer(out)
            out = self.activation_function(out)
            out = out + shortcut

        out = self.output_layer(out)
        return out

class Naisnet(nn.Module):
    def __init__(self, layers, activation):
        super(Naisnet, self).__init__()
        self.activation_function = activation
        self.epsilon = 0.01

        self.input_layer = nn.Linear(layers[0], layers[1])
        self.hidden_layers = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(1, len(layers) - 2)])
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self.input_layers = nn.ModuleList([nn.Linear(layers[0], layers[i]) for i in range(1, len(layers) - 1)])

    def project(self, layer, out):
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
            out = self.project(layer, out)
            out = out + self.input_layers[i](u)
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

class EulerMaruyamaScheme:
    #todo: seek out ways to improve the speed of MC here

    # def __init__(self, Xi, T, M, N, D, device):
    #     # self.Xi = torch.from_numpy(Xi).float().to(device)
    #     self.Xi = Xi.detach().cpu().numpy()
    #     # self.Xi.requires_grad = True
    #     self.T = T
    #     self.M = M
    #     self.N = N
    #     self.D = D
    #     self.device = device

    def __init__(self, Xi, T, M, N, D, device):
        self.Xi = Xi.to(device) if isinstance(Xi, torch.Tensor) else torch.from_numpy(Xi).float().to(device)
        # self.Xi = torch.from_numpy(Xi).float().to(device)
        self.Xi.requires_grad = True
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.device = device

    def simulate(self):
        Dt = np.zeros((self.M, self.N + 1, 1))
        DW = np.zeros((self.M, self.N + 1, self.D))

        dt = self.T / self.N
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(self.M, self.N, self.D))

        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W


class LossFunction:
    def __init__(self, model, g_tf, phi_tf, mu_tf, sigma_tf):
        self.model = model
        self.g_tf = g_tf
        self.phi_tf = phi_tf
        self.mu_tf = mu_tf
        self.sigma_tf = sigma_tf

    def compute(self, t, W, Xi):
        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi.repeat(self.model.M, 1).view(self.model.M, self.model.D)
        Y0, Z0 = self.model.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.model.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
                keepdim=True)
            Y1, Z1 = self.model.net_u(t1, X1)

            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            X_list.append(X0)
            Y_list.append(Y0)

        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.model.Dg_tf(X1), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0].item()


class NeuralNetworkTraining:
    def __init__(self, model, loss_function, simulator):
        self.model = model
        self.loss_function = loss_function
        self.simulator = simulator
        self.device = model.device

    def train(self, N_Iter, learning_rate):
        loss_temp = np.array([])

        previous_it = 0
        if self.model.iteration:
            previous_it = self.model.iteration[-1]

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        start_time = time.time()
        for it in range(previous_it, previous_it + N_Iter):
            self.optimizer.zero_grad()
            t_batch, W_batch = self.simulator.simulate()
            loss, X_pred, Y_pred, Y0_pred = self.loss_function.compute(t_batch, W_batch, self.simulator.Xi)
            loss.backward()
            self.optimizer.step()

            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            if it % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss.item(), Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            if it % 100 == 0:
                self.model.training_loss.append(loss_temp.mean())
                loss_temp = np.array([])
                self.model.iteration.append(it)

        return np.stack((self.model.iteration, self.model.training_loss))

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        loss, X_star, Y_star, Y0_pred = self.loss_function.compute(t_star, W_star, Xi_star)
        return X_star, Y_star


class FBSNN(ABC):
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        pass

class BlackScholesBarenblatt(nn.Module):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation, strike):
        super(BlackScholesBarenblatt, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Xi = torch.from_numpy(Xi).float().to(self.device)
        self.Xi.requires_grad = True
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.mode = mode
        self.activation = activation
        self.strike = strike * self.D # initialize the strike price

        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()
        elif activation == "Tanh":
            self.activation_function = nn.Tanh()

        if self.mode == "FC":
            self.layers = []
            for i in range(len(layers) - 2):
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                self.layers.append(self.activation_function)
            self.layers.append(nn.Linear(layers[-2], layers[-1]))
            self.model = nn.Sequential(*self.layers).to(self.device)
            # self.model = self.build_fc_model(layers)
        elif self.mode == "Resnet":
            self.model = Resnet(layers, self.activation_function)
        elif self.mode == "NAIS-Net":
            self.model = Naisnet(layers, self.activation_function)

        self.model.to(self.device)
        self.model.apply(self.weights_init)

        self.training_loss = []
        self.iteration = []

    # def build_fc_model(self, layers):
    #     fc_layers = []
    #     for i in range(len(layers) - 2):
    #         fc_layers.append(nn.Linear(layers[i], layers[i + 1]))
    #         fc_layers.append(self.activation_function)
    #     fc_layers.append(nn.Linear(layers[-2], layers[-1]))
    #     return nn.Sequential(*fc_layers)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):
        # t = t.detach().requires_grad_(True)
        # X = X.detach().requires_grad_(True)

        input = torch.cat((t, X), 1)
        u = self.model(input) # M x 1

        Du = torch.autograd.grad(outputs=u, inputs=X, grad_outputs=torch.ones_like(u), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]
        return u, Du

    def phi_tf(self, t, X, Y, Z):
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))

    def g_tf(self, X):
        temp = torch.sum(X, dim=1, keepdim=True)
        return torch.maximum(temp - self.strike, torch.tensor(0.0).to(self.device))

    def mu_tf(self, t, X, Y, Z):
        return torch.zeros([self.M, self.D]).to(self.device)

    def sigma_tf(self, t, X, Y):
        return 0.4 * torch.diag_embed(X)

    def Dg_tf(self, X):
        g = self.g_tf(X) # M x 1
        Dg = torch.autograd.grad(outputs= g, inputs= X, grad_outputs=torch.ones_like(g),
                                 allow_unused=True, retain_graph=True, create_graph=True)[0]
        return Dg


class CallOption(BlackScholesBarenblatt):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation, strike):
        super().__init__(Xi, T, M, N, D, layers, mode, activation, strike)

    def phi_tf(self, t, X, Y, Z):
        rate = 0.01  # Risk-free interest rate
        return rate * Y

    def g_tf(self, X):
        temp = torch.sum(X, dim=1, keepdim=True)
        return torch.maximum(temp - self.strike, torch.tensor(0.0).to(self.device))

    def mu_tf(self, t, X, Y, Z):
        rate = 0.01
        return rate * X

    def sigma_tf(self, t, X, Y):
        sigma = 0.25  # Volatility
        return sigma * torch.diag_embed(X)

    def Dg_tf(self, X):
        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=g, inputs=X, grad_outputs=torch.ones_like(g),
                                 allow_unused=True, retain_graph=True, create_graph=True)[0]
        return Dg


#
# def u_exact(t, X, r=0.05, sigma=0.4):
#     # This is used to calculate exact solution for European Call Option
#     d1 = (np.log(X) + (r + 0.5 * sigma**2) * (T - t)) / (sigma * np.sqrt(T - t))
#     d2 = d1 - sigma * np.sqrt(T - t) # this does not work because T-t could be 0
#     return X * normal.cdf(d1) - np.exp(-r * (T - t)) * normal.cdf(d2)


def black_scholes_call(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * np.exp(-q * T) * normal.cdf(d1)) - (K * np.exp(-r * T) * normal.cdf(d2))
    delta = normal.cdf(d1)
    return call_price, delta


def calculate_option_prices(X_pred, time_array, K, r, sigma, T, q=0):
    rows, cols = X_pred.shape
    option_prices = np.zeros((rows, cols))
    deltas = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            S = X_pred[i, j]
            t = time_array[j]
            time_to_maturity = T - t
            if time_to_maturity > 0:
                option_prices[i, j], deltas[i, j] = black_scholes_call(S, K, time_to_maturity, r, sigma, q)
            else:
                option_prices[i, j] = max(S - K, 0)
                if S > K:
                    deltas[i, j] = 1
                elif S == K:
                    deltas[i, j] = 0.5
                else:
                    deltas[i, j] = 0

    return option_prices, deltas


def run_model(model, N_Iter, learning_rate, save_path):
    tot = time.time()
    samples = 5
    print(model.device)

    # Training
    graph = trainer.train(N_Iter, learning_rate)
    print("total time:", time.time() - tot, "s")

    # Prediction
    np.random.seed(42)
    t_test, W_test = simulator.simulate()

    #todo: double check the functionality of the below two initializations

    # t_test = t_test.to(model.device)
    # W_test = W_test.to(model.device)

    #todo: double check the correctness of the simulator.Xi below
    X_pred, Y_pred = trainer.predict(simulator.Xi.cpu().detach().numpy(), t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    # for i in range(15):
    #     t_test_i, W_test_i = model.fetch_minibatch()
    #     X_pred_i, Y_pred_i = model.predict(Xi, t_test_i, W_test_i)
    #     if type(X_pred_i).__module__ != 'numpy':
    #         X_pred_i = X_pred_i.cpu().detach().numpy()
    #     if type(Y_pred_i).__module__ != 'numpy':
    #         Y_pred_i = Y_pred_i.cpu().detach().numpy()
    #     if type(t_test_i).__module__ != 'numpy':
    #         t_test_i = t_test_i.cpu().numpy()
    #     t_test = np.concatenate((t_test, t_test_i), axis=0)
    #     X_pred = np.concatenate((X_pred, X_pred_i), axis=0)
    #     Y_pred = np.concatenate((Y_pred, Y_pred_i), axis=0)
    X_pred = X_pred[:500, :]
    X_preds = X_pred[:, :, 0]

    # Calculate exact solution
    Y_test, Z_test = calculate_option_prices(X_preds, t_test[0], K, r = 0.05, sigma = 0.4, T = 1, q = 0)


    # Plotting
    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    training_loss_filename = f"{model.D}-dimensional-Call-Option-training-loss-{model.mode}-{model.activation}.png"
    training_loss_path = os.path.join(save_path, training_loss_filename)
    plt.savefig(training_loss_path)

    plt.figure()
    # plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    # plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    # plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')
    plt.plot(t_test[0] * 100, Y_pred[0] * 100, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0] * 100, Y_test[0] * 100, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0, -1] * 100, Y_test[0, -1] * 100, 'ko', label='$Y_T = u(T,X_T)$')

    # for i in range(1, samples):
    #     plt.plot(t_test[i:i+1, :, 0].T, Y_pred[i:i+1, :, 0].T, 'b')
    #     plt.plot(t_test[i:i+1, :, 0].T, Y_test[i:i+1, :, 0].T, 'r--')
    #     plt.plot(t_test[i:i+1, -1, 0], Y_test[i:i+1, -1, 0], 'ko')
    #
    # plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')
    for i in range(7):
        plt.plot(t_test[i] * 100, Y_pred[i] * 100, 'b')
        plt.plot(t_test[i] * 100, Y_test[i] * 100, 'r--')
        plt.plot(t_test[i, -1] * 100, Y_test[i, -1] * 100, 'ko')
    plt.plot([0], Y_test[0, 0] * 100, 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(f'{model.D}-dimensional Call Option, {model.mode}-{model.activation}')
    plt.legend()

    solution_plot_filename = f"{model.D}-dimensional-Call-Option-solution-{model.mode}-{model.activation}.png"
    solution_plot_path = os.path.join(save_path, solution_plot_filename)
    plt.savefig(solution_plot_path)

    # Calculate and plot relative errors
    # errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    # mean_errors = np.mean(errors, 0)
    # std_errors = np.std(errors, 0)
    #
    # plt.figure()
    # plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    # plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    # plt.xlabel('$t$')
    # plt.ylabel('relative error')
    # plt.title(f'{model.D}-dimensional Call Option, {model.mode}-{model.activation}')
    # plt.legend()
    #
    # error_plot_filename = f"{model.D}-dimensional-Call-Option-relative-error-{model.mode}-{model.activation}.png"
    # error_plot_path = os.path.join(save_path, error_plot_filename)
    # plt.savefig(error_plot_path)
    # Calculate and plot relative errors
    errors = np.abs((Y_test - Y_pred[:500, :, 0]) / Y_test)
    mean_errors = np.mean(errors, axis=0)
    std_errors = np.std(errors, axis=0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title(f'{model.D}-dimensional Call Option, {model.mode}-{model.activation}')
    plt.legend()

    error_plot_filename = f"{model.D}-dimensional-Call-Option-relative-error-{model.mode}-{model.activation}.png"
    error_plot_path = os.path.join(save_path, error_plot_filename)
    plt.savefig(error_plot_path)


if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions
    K = 1.0  # Strike price

    layers = [D + 1] + 4 * [256] + [1]
    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0

    mode = "NAIS-Net"  # FC, Resnet and NAIS-Net are available
    activation = "Sine"  # Sine, ReLU and Tanh are available

    save_path = r'C:\Users\aa04947\OneDrive - APG\Desktop\dnnpde_output'
    save_path = save_path.replace('\\', '/')  # Ensure correct formatting

    # Initialize Call Option model
    model = BlackScholesBarenblatt(Xi, T, M, N, D, layers, mode, activation, strike = K)

    # Initialize the EM simulator
    simulator = EulerMaruyamaScheme(model.Xi, model.T, model.M, model.N, model.D, model.device)

    # Initialize the loss function
    loss_function = LossFunction(model, model.g_tf, model.phi_tf, model.mu_tf, model.sigma_tf)

    # Initialize the trainer
    trainer = NeuralNetworkTraining(model, loss_function, simulator)

    # Run the model
    run_model(model, 10000, 1e-3, save_path)
