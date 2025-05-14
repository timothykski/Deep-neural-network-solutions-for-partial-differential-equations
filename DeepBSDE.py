import numpy as np
from abc import ABC, abstractmethod
import time
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

class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):

        device_idx = 0
        # The below code checks if GPU acceleration is available and potentially the oppo to move tensors to GPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx))
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        self.Xi = torch.from_numpy(Xi).float().to(self.device) # This line code moves tensors to GPU
        self.Xi.requires_grad = True

        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.mode = mode
        self.activation = activation

        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()

        if self.mode == "FC":
            self.layers = []
            for i in range(len(layers) - 2):
                self.layers.append(nn.Linear(layers[i], layers[i + 1]))
                self.layers.append(self.activation_function)
            self.layers.append(nn.Linear(layers[-2], layers[-1]))
            self.model = nn.Sequential(*self.layers).to(self.device)
        elif self.mode in ["NAIS-Net", "Resnet"]:
            self.model = Resnet(layers, stable=(self.mode == "NAIS-Net"), activation=self.activation_function).to(self.device)
        elif self.mode == "Verlet":
            self.model = VerletNet(layers, activation=self.activation_function).to(self.device)
        elif self.mode == "SDEnet":
            self.model = SDEnet(layers, activation=self.activation_function).to(self.device)

        self.model.apply(self.weights_init)

        self.training_loss = []
        self.iteration = []

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):
        input = torch.cat((t, X), 1)
        u = self.model(input)
        Du = torch.autograd.grad(outputs=u, inputs=X, grad_outputs=torch.ones_like(u), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]
        return u, Du

    def Dg_tf(self, X):
        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=g, inputs=X, grad_outputs=torch.ones_like(g), allow_unused=True,
                                 retain_graph=True, create_graph=True)[0]
        return Dg

    def loss_function(self, t, W, Xi):
        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)
        Y0, Z0 = self.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)


        # Euler scheme: the trajectory of X_n and the corresponding Y_n are simulated
        # Euler Maruyama method for approximating Xn, Yn and Zn
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

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            X_list.append(X0)
            Y_list.append(Y0)

        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)

        return loss, X, Y, Y[0, 0, 0].item()

    def fetch_minibatch(self):
        # The training set is implicitly generated through this method (simulates trajectories of SDE)
        Dt = np.zeros((self.M, self.N + 1, 1))
        DW = np.zeros((self.M, self.N + 1, self.D))

        #todo: the below step has room to implement multi-level Monte Carlo Simulation
        dt = self.T / self.N
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(self.M, self.N, self.D))

        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)

        return t, W

    # This is where Neural Network is trained
    def train(self, N_Iter, learning_rate):
        loss_temp = np.array([])

        previous_it = 0
        if self.iteration:
            previous_it = self.iteration[-1]

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        start_time = time.time()
        for it in range(previous_it, previous_it + N_Iter):
            self.optimizer.zero_grad()
            t_batch, W_batch = self.fetch_minibatch()
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            loss.backward()
            self.optimizer.step()

            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())

            if it % 100 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                      (it, loss.item(), Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            if it % 100 == 0:
                self.training_loss.append(loss_temp.mean())
                loss_temp = np.array([])
                self.iteration.append(it)

        return np.stack((self.iteration, self.training_loss))

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        loss, X_star, Y_star, Y0_pred = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        # This is the final condition
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)


class BlackScholesBarenblatt(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return 0.05 * (Y - torch.sum(X * Z, dim=1, keepdim=True))  # M x 1

    def g_tf(self, X):  # M x D
        # Final condition
        return torch.sum(X ** 2, 1, keepdim=True)  # M x 1

    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        return super().mu_tf(t, X, Y, Z)  # M x D

    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        return 0.4 * torch.diag_embed(X)  # M x D x D



def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    # This is used to calculate exact solution
    r = 0.05
    sigma_max = 0.4
    return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1


def run_model(model, N_Iter, learning_rate):
    tot = time.time()
    # Define the local directory where you want to save the plot
    save_path = r'C:\Users\aa04947\OneDrive - APG\Desktop\dnnpde_output'
    save_path = save_path.replace('\\', '/')  # Ensure correct formatting

    samples = 5
    print(model.device)
    graph = model.train(N_Iter, learning_rate)
    print("total time:", time.time() - tot, "s")

    np.random.seed(42)
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')

    # Save training loss plot
    training_loss_filename = f"{D}-dimensional-Black-Scholes-Barenblatt-training-loss-{model.mode}-{model.activation}.png"
    training_loss_path = f"{save_path}/{training_loss_filename}"
    plt.savefig(training_loss_path)

    plt.figure()
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()

    # Save learned vs exact solution plot
    solution_plot_filename = f"{D}-dimensional-Black-Scholes-Barenblatt-solution-{model.mode}-{model.activation}.png"
    solution_plot_path = f"{save_path}/{solution_plot_filename}"
    plt.savefig(solution_plot_path)

    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure()
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')

    plt.title(str(D) + '-dimensional Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()

    # Save relative error plot
    error_plot_filename = f"{D}-dimensional-Black-Scholes-Barenblatt-relative-error-{model.mode}-{model.activation}.png"
    error_plot_path = f"{save_path}/{error_plot_filename}"
    plt.savefig(error_plot_path)
    plt.show()


if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time steps
    D = 100  # number of dimensions

    layers = [D + 1] + 4 * [256] + [1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :] # todo: what is this?
    T = 1.0

    "Available architectures"
    mode = "FC"  # FC: Fully Connected, Resnet and NAIS-Net are available
    activation = "Sine"  # sine and ReLU are available #todo: check other methods of calling activation
    model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D,
                                   layers, mode, activation)
    # run_model(model, 2*10**4, 1e-3)
    run_model(model, 100, 1e-3)