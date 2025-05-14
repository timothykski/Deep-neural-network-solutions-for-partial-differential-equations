import numpy as np
import torch
from scipy.stats import norm


class CallOption:
    def __init__(self, Xi, T, M, N, D, r=0.05, sigma=0.2):
        self.Xi = Xi
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.r = r
        self.sigma = sigma

    def simulate_asset_paths(self):
        dt = self.T / self.N
        paths = np.zeros((self.M, self.D, self.N + 1))
        paths[:, :, 0] = self.Xi

        for t in range(1, self.N + 1):
            z = np.random.standard_normal((self.M, self.D))
            paths[:, :, t] = paths[:, :, t - 1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z
            )

        return paths

    def phi(self, t, X, Y, Z):
        avg_XZ = torch.sum(X * Z, dim=1, keepdim=True) / self.D
        return self.r * (Y - avg_XZ)

    def g(self, X):
        avg_X = torch.mean(X, dim=1, keepdim=True)
        return torch.maximum(avg_X - 1.0, torch.tensor(0.0))  # Payoff for the basket option

    def mu(self, t, X, Y, Z):
        return self.r * X

    def sigma(self, t, X, Y):
        return self.sigma * torch.diag_embed(X)

    def price_option(self):
        paths = self.simulate_asset_paths()
        final_values = paths[:, :, -1]
        payoffs = self.g(torch.tensor(final_values, dtype=torch.float32))
        discounted_payoff = torch.exp(-self.r * self.T) * payoffs
        option_price = torch.mean(discounted_payoff)
        return option_price.item()

# Option price calculator class for comparison
class OptionPriceCalculator:
    @staticmethod
    def black_scholes_call(S, K, T, r, sigma, dimensions, q=0):
        sigma_avg = sigma / np.sqrt(dimensions)
        S_avg = np.mean(S)
        d1 = (np.log(S_avg / K) + (r + 0.5 * sigma_avg ** 2) * T) / (sigma_avg * np.sqrt(T))
        d2 = d1 - sigma_avg * np.sqrt(T)
        call_price = S_avg * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
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

                t = time_array[min(j, len(time_array) - 1)]  # Ensure correct indexing
                time_to_maturity = T - t
                if time_to_maturity > 0:
                    option_prices[i, j], deltas[i, j] = self.black_scholes_call(avg_S, K, time_to_maturity, r, sigma, dimensions, q)
                else:
                    option_prices[i, j] = max(avg_S - K, 0)
                    if avg_S > K:
                        deltas[i, j] = 1
                    elif avg_S == K:
                        deltas[i, j] = 0.5
                    else:
                        deltas[i, j] = 0

        return option_prices, deltas

# Example usage
if __name__ == "__main__":
    dimensions = 2  # Number of assets
    S0 = np.ones(dimensions)  # Initial stock prices
    K = 1.0  # Strike price
    T = 1  # Time to maturity in years
    r = 0.05  # Risk-free interest rate
    sigma = 0.2  # Volatility
    M = 10000  # Number of Monte Carlo simulations
    N = 50 # Number of time steps

    call_option = CallOption(S0, T, M, N, dimensions, r, sigma)
    mc_price = call_option.price_option()
    print(f"Basket Option Price (MC): {mc_price:.4f}")

    # Option price calculator for analytical comparison
    option_calculator = OptionPriceCalculator()
    X_pred = np.array([S0 for _ in range(M)]).reshape(M, 1, dimensions)
    time_array = np.linspace(0, T, N + 1)
    analytical_prices, _ = option_calculator.calculate_call_option_prices(X_pred, time_array, K, r, sigma, T, dimensions)
    analytical_price = np.mean(analytical_prices[:, 0])
    print(f"Basket Option Price (Analytical): {analytical_price:.4f}")
