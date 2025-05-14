import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import math


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

        plt.show()

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

    def simulate_paths(self):
        S = np.zeros((self.num_steps + 1, self.num_paths))
        S[0, :] = self.S0
        V = np.zeros((self.num_steps + 1, self.num_paths))
        V[0, :] = self.v0
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

    def price_option(self):
        S, V = self.simulate_paths()
        payoff = np.maximum(S[-1, :] - self.K, 0)  # Call option payoff
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        std_err = np.std(payoff) / np.sqrt(self.num_paths)
        return price, std_err



    def plot_paths(self):
        S, V = self.simulate_paths()
        plt.figure(figsize=(12, 6))
        plt.plot(S[:, :10])  # Plotting the first 10 paths for clarity
        plt.title('Simulated Asset Paths')
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Price')
        plt.show()

# Use the new class in the main execution
if __name__ == "__main__":
    # Parameters for Heston model and Monte Carlo
    S0 = 100
    K = 100
    r = 0.05
    T = 1
    kappa = 2 # mean reversion rate
    theta = 0.2 # long term mean of variance
    sigma = 0.3 # volatility of volatility
    rho = 0.8 # correlation between Brownian motions
    v0 = 0.2 # initial variance
    Smax = 200  # Maximum asset price
    vmax = 1  # Maximum variance
    M = 40  # Number of asset price steps
    N = 20  # Number of volatility steps

    S_values = np.linspace(0, Smax, M + 1)
    V_values = np.linspace(0, vmax, N + 1)

    heston_surface = HestonClosedFormSurface(S0, K, r, T, kappa, theta, sigma, rho, v0)

    # Print the option price at the initial spot price and volatility
    initial_price = heston_surface.call_price(S0, v0)
    print(f"Heston Model Call Option Price at S0={S0}, v0={v0}: {initial_price:.4f}")

    # Plot the surfaces
    heston_surface.plot_surfaces(S_values, V_values)

    num_paths = 20000
    num_simulations = 20000
    num_time_steps = 100

    # Instantiate and price using Monte Carlo (New)
    mc_pricer = MonteCarloPricer(S0, K, r, T, kappa, theta, sigma, rho, v0, num_paths)
    mc_price, mc_std_err = mc_pricer.price_option()
    print(f"Monte Carlo Simulation Call Option Price: {mc_price:.4f} ± {mc_std_err:.4f}")

    # Optional: Plot simulated paths
    mc_pricer.plot_paths()
