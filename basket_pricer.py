import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time


class MonteCarloSimulator:
    def __init__(self, S0, r, sigma, T, dt, correlation_matrix=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.correlation_matrix = correlation_matrix
        self.num_assets = len(S0)
        self.num_steps = int(T / dt)

        if correlation_matrix is not None:
            self.correlation_matrix = self._make_positive_definite(correlation_matrix)
            self.L = np.linalg.cholesky(self.correlation_matrix)
        else:
            self.L = np.eye(self.num_assets)

    def _is_positive_definite(self, matrix):
        """Check if a matrix is positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def _make_positive_definite(self, matrix):
        """Adjust the matrix to make it positive definite if necessary."""
        epsilon = 1e-10
        adjusted_matrix = matrix
        while not self._is_positive_definite(adjusted_matrix):
            adjusted_matrix += epsilon * np.eye(matrix.shape[0])
            epsilon *= 10
        return adjusted_matrix

    def simulate(self, num_simulations):
        # Vectorized implementation for generating asset paths
        Z = np.random.normal(size=(self.num_assets, self.num_steps, num_simulations))
        correlated_Z = np.tensordot(self.L, Z, axes=(1, 0))

        time_grid = np.linspace(0, self.T, self.num_steps + 1)
        drift = (self.r - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * correlated_Z

        asset_paths = np.exp(drift + diffusion)
        asset_paths = np.cumprod(np.insert(asset_paths, 0, self.S0[:, np.newaxis], axis=1), axis=1)

        return asset_paths


class BasketOptionPricer:
    def __init__(self, strike, T, correlation_matrix=None):
        self.strike = strike
        self.T = T
        self.correlation_matrix = correlation_matrix

    def price(self, asset_paths, r):
        avg_asset_prices = np.mean(asset_paths, axis=0)
        payoff = np.maximum(avg_asset_prices[-1, :] - self.strike, 0)
        option_price = np.exp(-r * self.T) * np.mean(payoff)
        return option_price

    def delta(self, S0, asset_paths, r, sigma, T, dt, epsilon=1e-4):
        num_assets = len(S0)
        deltas = np.zeros(num_assets)
        option_price_base = self.price(asset_paths, r)

        for i in range(num_assets):
            S0_perturbed = S0.copy()
            S0_perturbed[i] += epsilon
            perturbed_simulator = MonteCarloSimulator(S0_perturbed, r, sigma, T, dt, self.correlation_matrix)
            perturbed_asset_paths = perturbed_simulator.simulate(asset_paths.shape[2])
            perturbed_option_price = self.price(perturbed_asset_paths, r)
            deltas[i] = (perturbed_option_price - option_price_base) / epsilon

        return deltas

    def price_and_delta(self, S0, asset_paths, r, sigma, T, dt):
        option_price = self.price(asset_paths, r)
        deltas = self.delta(S0, asset_paths, r, sigma, T, dt)
        return option_price, deltas


class RandomCorrelationSensitivityAnalysis:
    def __init__(self, n, r, sigma, T, dt, K):
        self.n = n
        self.S0 = np.ones(n)  # Initial prices
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.K = K

    def generate_positive_definite_correlation_matrix(self, restrict_positive=False):
        """Generates a random positive definite correlation matrix."""
        random_matrix = np.random.randn(self.n, self.n)
        if restrict_positive:
            random_matrix = np.abs(random_matrix)  # Ensure all correlations are positive
        random_corr_matrix = np.dot(random_matrix, random_matrix.T)
        np.fill_diagonal(random_corr_matrix, 1)
        d = np.sqrt(np.diag(random_corr_matrix))
        random_corr_matrix = random_corr_matrix / np.outer(d, d)

        # Ensure positive definiteness
        while not self.is_positive_definite(random_corr_matrix):
            random_matrix = np.random.randn(self.n, self.n)
            if restrict_positive:
                random_matrix = np.abs(random_matrix)  # Ensure all correlations are positive
            random_corr_matrix = np.dot(random_matrix, random_matrix.T)
            np.fill_diagonal(random_corr_matrix, 1)
            d = np.sqrt(np.diag(random_corr_matrix))
            random_corr_matrix = random_corr_matrix / np.outer(d, d)

        return random_corr_matrix

    def is_positive_definite(self, matrix):
        """Check if the matrix is positive definite."""
        try:
            np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError:
            return False

    def run_analysis(self, correlation_type="no_correlation", num_simulations=10000, num_samples=100):
        option_prices = []
        correlation_matrices = []
        for _ in range(num_samples):
            if correlation_type == "random_correlation":
                correlation_matrix = self.generate_positive_definite_correlation_matrix(restrict_positive=False)
            elif correlation_type == "restricted_random_correlation":
                correlation_matrix = self.generate_positive_definite_correlation_matrix(restrict_positive=True)
            else:
                correlation_matrix = None

            if correlation_matrix is not None:
                correlation_matrices.append(correlation_matrix.flatten())

            analysis = OptionPricingAnalysis(self.S0, self.r, self.sigma, self.T, self.dt, self.K, correlation_matrix)
            _, option_price = analysis.run_analysis(num_simulations)
            option_prices.append(option_price)

        return option_prices, correlation_matrices

    def plot_pca_results(self, option_prices, correlation_matrices, n_components=2):
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(correlation_matrices)

        plt.figure(figsize=(10, 6))
        if n_components == 1:
            plt.scatter(principal_components[:, 0], option_prices, c='blue', edgecolor='black')
            plt.xlabel('Principal Component 1')
        elif n_components == 2:
            plt.scatter(principal_components[:, 0], principal_components[:, 1], c=option_prices, cmap='viridis',
                        edgecolor='black')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.colorbar(label='Option Price')
        plt.title(f'PCA Analysis: Option Prices vs. Principal Components (n={self.n})')
        plt.grid(True)
        plt.show()

class OptionPricingAnalysis:
    def __init__(self, S0, r, sigma, T, dt, K, correlation_matrix=None):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.K = K
        self.correlation_matrix = correlation_matrix

    def run_analysis(self, num_simulations=10000):
        simulator = MonteCarloSimulator(self.S0, self.r, self.sigma, self.T, self.dt, self.correlation_matrix)
        asset_paths = simulator.simulate(num_simulations)

        pricer = BasketOptionPricer(self.K, self.T)
        option_price = pricer.price(asset_paths, self.r)

        return asset_paths, option_price

    def plot_asset_paths(self, asset_paths, title):
        plt.figure(figsize=(10, 6))
        for i in range(asset_paths.shape[0]):
            plt.plot(asset_paths[i, :, :10])  # Plot first 10 simulated paths for each asset
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Asset Prices')
        plt.show()

    def plot_3d_option_price(self, asset_paths, option_price, title):
        spot_prices = np.mean(asset_paths[:, -1, :], axis=0)
        time_to_maturity = np.linspace(0, self.T, asset_paths.shape[1])

        X, Y = np.meshgrid(spot_prices, time_to_maturity)
        Z = np.tile(option_price, (len(time_to_maturity), 1))

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_title(title)
        ax.set_xlabel('Spot Prices')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel('Option Price')
        plt.show()

    def sensitivity_analysis(self, correlation_range, num_simulations=10000):
        option_prices = []
        for rho in correlation_range:
            correlation_matrix = np.full((len(self.S0), len(self.S0)), rho)
            np.fill_diagonal(correlation_matrix, 1.0)

            self.correlation_matrix = correlation_matrix
            _, option_price = self.run_analysis(num_simulations)
            option_prices.append(option_price)

        plt.figure(figsize=(10, 6))
        plt.plot(correlation_range, option_prices, marker='o')
        plt.title('Sensitivity of Option Price to Correlation')
        plt.xlabel('Correlation')
        plt.ylabel('Option Price')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    start_time = time.time()  # Start measuring time

    # Parameters
    n = 5  # Number of assets
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0  # Time to maturity
    dt = 0.01  # Time step
    K = 1.0  # Strike price
    S0 = np.ones(n)  # Initial prices

    # Choose the type of correlation
    correlation_type = "random_correlation"  # Options: "no_correlation", "random_correlation", "restricted_random_correlation"

    # Perform analysis based on the selected correlation type
    random_corr_analysis = RandomCorrelationSensitivityAnalysis(n, r, sigma, T, dt, K)
    option_prices, correlation_matrices = random_corr_analysis.run_analysis(correlation_type=correlation_type)

    # Extract the first correlation matrix used in the analysis
    correlation_matrix = None
    if correlation_matrices:
        correlation_matrix = np.reshape(correlation_matrices[0], (n, n))

    # Perform option pricing and delta calculation
    simulator = MonteCarloSimulator(S0, r, sigma, T, dt, correlation_matrix)
    asset_paths = simulator.simulate(num_simulations=10000)

    pricer = BasketOptionPricer(K, T, correlation_matrix)
    option_price, deltas = pricer.price_and_delta(S0, asset_paths, r, sigma, T, dt)

    print(f"Option Price: {option_price}")
    print(f"Delta: {deltas}")

    # Plot the PCA results
    # random_corr_analysis.plot_pca_results(option_prices, correlation_matrices, n_components=2)

    end_time = time.time()  # End measuring time
    execution_time = end_time - start_time  # Calculate the execution time
    print(f"\nTime taken to execute the algorithm: {execution_time:.2f} seconds")
