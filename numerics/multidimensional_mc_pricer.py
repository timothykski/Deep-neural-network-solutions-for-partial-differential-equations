import numpy as np
from scipy.stats import norm
from scipy.fftpack import fft
from scipy.interpolate import interp1d


class CorrelationMatrix:
    def __init__(self, dimensions, with_correlation=True):
        self.dimensions = dimensions
        self.with_correlation = with_correlation
        self.matrix = self.generate_correlation_matrix()

    def generate_correlation_matrix(self):
        if self.with_correlation:
            return self.generate_positive_definite_correlation_matrix()
        else:
            return np.eye(self.dimensions)  # Identity matrix for no correlation

    def generate_positive_definite_correlation_matrix(self):
        """
        Generates a random positive definite correlation matrix.
        """
        # Generate a random matrix
        A = np.random.rand(self.dimensions, self.dimensions)

        # Symmetrize the matrix
        A = 0.5 * (A + A.T)

        # Add dimensions * identity matrix to ensure positive definiteness
        A += self.dimensions * np.eye(self.dimensions)

        # Normalize to get a correlation matrix
        D_inv = np.diag(1.0 / np.sqrt(np.diag(A)))
        correlation_matrix = D_inv @ A @ D_inv

        return correlation_matrix


class BlackScholesModel:
    def __init__(self, rate, sigma, dimensions, with_correlation=False):
        self.rate = rate
        self.sigma = sigma
        self.dimensions = dimensions
        self.with_correlation = with_correlation

        self.correlation_matrix = CorrelationMatrix(dimensions, with_correlation)
        self.correlation = self.correlation_matrix.matrix

    def generate_paths(self, S0, T, N, num_simulations):
        dt = T / N
        paths = np.zeros((num_simulations, N + 1, self.dimensions))
        paths[:, 0, :] = S0

        if self.with_correlation:
            L = np.linalg.cholesky(self.correlation)
        else:
            L = np.eye(self.dimensions)  # No correlation, use identity matrix

        for t in range(1, N + 1):
            z = np.random.standard_normal((num_simulations, self.dimensions))
            z = z @ L.T

            paths[:, t, :] = paths[:, t - 1, :] * np.exp(
                (self.rate - 0.5 * self.sigma ** 2) * dt + self.sigma * np.sqrt(dt) * z
            )

        return paths


class BasketOption:
    def __init__(self, weights, strike):
        self.weights = weights
        self.strike = strike

    def payoff(self, S):
        basket_price = np.sum(S * self.weights, axis=1)
        return np.maximum(basket_price - self.strike, 0)


class MonteCarloPricer:
    def __init__(self, model, option, T, N, num_simulations):
        self.model = model
        self.option = option
        self.T = T
        self.N = N
        self.num_simulations = num_simulations

    def price(self, S0):
        paths = self.model.generate_paths(S0, self.T, self.N, self.num_simulations)
        terminal_prices = paths[:, -1, :]
        payoffs = self.option.payoff(terminal_prices)
        discounted_payoff = np.exp(-self.model.rate * self.T) * payoffs
        return np.mean(discounted_payoff)


class AnalyticalBlackScholes:
    def __init__(self, rate, sigma, dimensions):
        self.rate = rate
        self.sigma = sigma
        self.dimensions = dimensions
        self.sigma_avg = sigma / np.sqrt(dimensions)

    def price(self, S0, strike, T):
        S_avg = np.mean(S0)
        d1 = (np.log(S_avg / strike) + (self.rate + 0.5 * self.sigma_avg ** 2) * T) / (self.sigma_avg * np.sqrt(T))
        d2 = d1 - self.sigma_avg * np.sqrt(T)
        call_price = S_avg * norm.cdf(d1) - strike * np.exp(-self.rate * T) * norm.cdf(d2)
        return call_price


class FFTPricer:
    def __init__(self, model, option, T):
        self.model = model
        self.option = option
        self.T = T

    def characteristic_function(self, u, S0, weights):
        """
        Calculate the characteristic function of the log of the basket price under the risk-neutral measure.
        u: Integration variable for the characteristic function (complex number)
        S0: Initial prices of the assets
        weights: Weights of the assets in the basket
        """
        rate = self.model.rate
        sigma = self.model.sigma
        correlation = self.model.correlation
        dimensions = self.model.dimensions

        # Compute weighted average initial price
        log_S0 = np.log(S0)
        avg_S0 = np.dot(weights, log_S0)

        # Compute the variance of the basket
        variance = 0
        for i in range(dimensions):
            for j in range(dimensions):
                variance += weights[i] * weights[j] * sigma * sigma * correlation[i, j]

        variance = variance * self.T
        mean = avg_S0 + (rate - 0.5 * variance) * self.T

        # Characteristic function
        cf_value = np.exp(1j * u * mean - 0.5 * variance * u ** 2)
        return cf_value

    def fft_option_price(self, S0, weights, K, N=2 ** 12, alpha=1.5):
        """
        Compute the price of the option using the Fast Fourier Transform (FFT).
        S0: Initial prices of the assets
        weights: Weights of the assets in the basket
        K: Strike price of the option
        N: Number of FFT points
        alpha: Damping factor to ensure convergence
        """
        # Set up grid for FFT
        log_strike_grid = np.linspace(-10, 10, N)
        strikes = np.exp(log_strike_grid)

        # Compute the characteristic function values
        delta_k = log_strike_grid[1] - log_strike_grid[0]
        u = np.fft.fftfreq(N, d=delta_k)
        u = 2 * np.pi * u
        cf_values = self.characteristic_function(u - (alpha + 1) * 1j, S0, self.option.weights)

        # FFT step
        payoff_cf = np.exp(-self.model.rate * self.T) * cf_values * np.exp(1j * u * log_strike_grid[0]) / (
                    alpha ** 2 + alpha - u ** 2 + 1j * (2 * alpha + 1) * u)
        prices_fft = fft(payoff_cf).real
        prices = prices_fft / np.pi

        # Interpolation to find the option price for the given strike
        interpolation = interp1d(strikes, prices, kind='linear')
        price_at_strike = interpolation(K)
        return price_at_strike

    def price(self, S0):
        # Calculate the FFT price for the option
        strike = self.option.strike
        weights = self.option.weights
        price = self.fft_option_price(S0, weights, strike)
        return price


class CentralMomentPricer:
    def __init__(self, model, option, T):
        self.model = model
        self.option = option
        self.T = T

    def compute_moments(self, S0):
        """
        Compute the first three moments of the basket price at maturity.
        """
        n = self.model.dimensions
        weights = self.option.weights
        sigma = self.model.sigma
        rate = self.model.rate

        # Access the actual correlation matrix
        corr_matrix = self.model.correlation_matrix.matrix

        # First moment (mean)
        mu_B = np.sum(weights * S0 * np.exp(rate * self.T))

        # Second moment
        second_moment_sum = 0
        for i in range(n):
            for j in range(n):
                second_moment_sum += weights[i] * weights[j] * S0[i] * S0[j] * \
                                     np.exp(rate * 2 * self.T) * \
                                     np.exp(corr_matrix[i, j] * sigma ** 2 * self.T)
        sigma_B = np.sqrt(max(second_moment_sum - mu_B ** 2, 0))  # Ensure non-negative

        # Third moment
        third_moment_sum = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    third_moment_sum += weights[i] * weights[j] * weights[k] * S0[i] * S0[j] * S0[k] * \
                                        np.exp(rate * 3 * self.T) * \
                                        np.exp((corr_matrix[i, j] +
                                                corr_matrix[i, k] +
                                                corr_matrix[j, k]) * sigma ** 2 * self.T)
        skew_B = (third_moment_sum - 3 * mu_B * sigma_B ** 2 - mu_B ** 3) / max(sigma_B ** 3, 1e-10)

        return mu_B, sigma_B, skew_B

    def match_moments(self, mu_B, sigma_B, skew_B):
        """
        Match the moments to find c, s, m, and τ.
        """
        x = np.sign(skew_B) * np.abs(skew_B / 2) ** (1 / 3)

        # Handle invalid cases for x
        if x <= 1:
            x = 1.01  # Adjust x slightly above 1 to avoid issues

        c = np.sign(skew_B)
        s = (sigma_B / np.sqrt(np.abs(x * (x - 1)))) ** (1 / 2) if x > 1 else 0
        m = 0.5 * np.log(np.abs(x * (x - 1))) + np.log(mu_B) if x > 1 else np.log(mu_B)
        tau = c * sigma_B * mu_B * np.sqrt(np.abs(x - 1)) if x > 1 else 0

        return c, s, m, tau

    def price(self, S0):
        """
        Calculate the price of the basket option using the central moments method.
        """
        mu_B, sigma_B, skew_B = self.compute_moments(S0)
        c, s, m, tau = self.match_moments(mu_B, sigma_B, skew_B)
        K = self.option.strike

        # Calculate d1 and d2 based on the moments
        d1 = (np.log(mu_B / K) + (s ** 2) / 2) / max(s, 1e-10)
        d2 = d1 - s

        # Price using the derived parameters
        if c == 1 and K <= tau:
            price = np.exp(-self.model.rate * self.T) * (mu_B * norm.cdf(d1) - K * norm.cdf(d2))
        elif c == 1 and K > tau:
            price = np.exp(-self.model.rate * self.T) * (mu_B * norm.cdf(d1) - K * norm.cdf(d2))
        elif c == -1 and K >= -tau:
            price = np.exp(-self.model.rate * self.T) * (mu_B * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = np.exp(-self.model.rate * self.T) * (mu_B * norm.cdf(d1) - K * norm.cdf(d2))

        return price



def main():
    rate = 0.05
    sigma = 0.20
    dimensions = 5  # Define the dimension parameter D
    T = 1.0
    N = 50
    num_simulations = 20000  # You can adjust the number of simulations

    with_correlation = True  # Set this to False if you want to turn off correlation

    S0 = np.ones(dimensions)  # Initial price of each asset
    weights = np.ones(dimensions) / dimensions  # Equal weights for each asset
    strike = 1.0  # Example strike price

    model = BlackScholesModel(rate, sigma, dimensions, with_correlation)
    option = BasketOption(weights, strike)
    pricer = MonteCarloPricer(model, option, T, N, num_simulations)

    analytical_model = AnalyticalBlackScholes(rate, sigma, dimensions)
    analytical_price = analytical_model.price(S0, strike, T) * 100
    print(f"The analytical price of the basket option without correlation is: {analytical_price:.2f}")

    price = pricer.price(S0) * 100  # This 100 is only for scaling to make the price look more realistic
    print(f"The Monte Carlo price of the basket option is: {price:.2f}")

    fft_pricer = FFTPricer(model, option, T)
    fft_price = fft_pricer.price(S0) * 100
    print(f"The FFT-based price of the basket option is: {fft_price:.2f}")

    central_moment_pricer = CentralMomentPricer(model, option, T)
    price = central_moment_pricer.price(S0)*100
    print(f"The Central Moments price of the basket option is: {price:.2f}")


if __name__ == "__main__":
    main()
