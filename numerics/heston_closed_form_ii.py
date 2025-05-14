import numpy as np
import matplotlib.pyplot as plt
import math


class MonteCarloPricer:
    def __init__(self, S0, K, r, T, kappa, theta, sigma, rho, v0, num_paths, dt=0.001):
        self.S0 = S0
        self.K = K
        self.r = r
        self.T = T
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.v0 = v0
        self.num_paths = num_paths
        self.dt = dt
        self.num_steps = int(T / dt)

    def simulate_paths(self, S_initial, V_initial):
        S = np.zeros((self.num_steps + 1, self.num_paths))
        S[0, :] = S_initial
        V = np.zeros((self.num_steps + 1, self.num_paths))
        V[0, :] = V_initial
        for i in range(self.num_paths):
            for t_step in range(1, self.num_steps + 1):
                Zv = np.random.randn(1)
                Zs = self.rho * Zv + math.sqrt(1 - self.rho ** 2) * np.random.randn(1)

                # Milstein scheme for V
                V[t_step, i] = V[t_step - 1, i] + self.kappa * (self.theta - V[t_step - 1, i]) * self.dt + \
                               self.sigma * math.sqrt(V[t_step - 1, i]) * math.sqrt(self.dt) * Zv + \
                               0.25 * self.sigma ** 2 * self.dt * (Zv ** 2 - 1)

                # Ensure V does not go negative
                V[t_step, i] = max(V[t_step, i], 0)

                # Simulate S using V
                S[t_step, i] = S[t_step - 1, i] * np.exp((self.r - V[t_step - 1, i] / 2) * self.dt + \
                                                         math.sqrt(V[t_step - 1, i]) * math.sqrt(self.dt) * Zs)

        return S, V

    def price_option(self, S_initial, V_initial):
        S, V = self.simulate_paths(S_initial, V_initial)
        payoff = np.maximum(S[-1, :] - self.K, 0)  # Call option payoff
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        std_err = np.std(payoff) / np.sqrt(self.num_paths)
        return price

    def price_surface(self, S_values, V_values):
        price_grid = np.zeros((len(S_values), len(V_values)))

        for i, S in enumerate(S_values):
            for j, V in enumerate(V_values):
                price_grid[i, j] = self.price_option(S, V)

        return price_grid

    def delta_surface(self, S_values, V_values):
        delta_grid = np.zeros((len(S_values), len(V_values)))
        dS = S_values[1] - S_values[0]

        for i in range(len(S_values) - 1):
            for j in range(len(V_values)):
                delta_grid[i, j] = (self.price_option(S_values[i + 1], V_values[j]) - self.price_option(S_values[i],
                                                                                                       V_values[j])) / dS

        return delta_grid

    def gamma_surface(self, S_values, V_values):
        gamma_grid = np.zeros((len(S_values), len(V_values)))
        dS = S_values[1] - S_values[0]

        for i in range(1, len(S_values) - 1):
            for j in range(len(V_values)):
                gamma_grid[i, j] = (self.price_option(S_values[i + 1], V_values[j]) - 2 * self.price_option(S_values[i],
                                                                                                            V_values[
                                                                                                                j]) + self.price_option(
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

        plt.show()


# Use the new class in the main execution
if __name__ == "__main__":
    # Parameters for Heston model and Monte Carlo
    S0 = 100
    K = 100
    r = 0.05
    T = 1
    kappa = 2
    theta = 0.2
    sigma = 0.3
    rho = 0.8
    v0 = 0.2
    Smax = 200  # Maximum asset price
    vmax = 1  # Maximum variance
    M = 40  # Number of asset price steps
    N = 20  # Number of volatility steps

    S_values = np.linspace(0, Smax, M + 1)
    V_values = np.linspace(0, vmax, N + 1)

    # Instantiate and price using Monte Carlo
    mc_pricer = MonteCarloPricer(S0, K, r, T, kappa, theta, sigma, rho, v0, num_paths=2000)

    # Plot the surfaces
    mc_pricer.plot_surfaces(S_values, V_values)
