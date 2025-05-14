import numpy as np
from scipy.integrate import quad
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class AssetParameters:
    S0: float
    weight: float
    sigma: float

class VGBasketOptionPricer:
    def __init__(self, assets: List[AssetParameters], K: float, T: float, r: float, sigma: float, nu: float, theta: float, rho: float = None):
        self.assets = assets
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.nu = nu
        self.theta = theta
        self.rho = rho
        self.n = len(assets)

    def price(self) -> float:
        return self._price_vg(self.sigma, self.nu, self.theta)

    def _price_vg(self, sigma: float, nu: float, theta: float) -> float:
        S0 = sum(asset.S0 * asset.weight for asset in self.assets)
        omega = (1 / nu) * np.log(max(1e-10, 1 - theta * nu - 0.5 * sigma**2 * nu))

        def integrand(u):
            cf = self._vg_characteristic_function(u, sigma, nu, theta, omega, S0)
            return np.real(np.exp(-1j * u * np.log(self.K)) * cf / (1j * u))

        #todo: something is still wrong in the calculation
        integral, _ = quad(integrand, 0, 1000, limit=3000, epsabs=1e-9, epsrel=1e-9)
        price = S0 - self.K * np.exp(-self.r * self.T) * (0.5 + 1/np.pi * integral)
        return max(price, 0)

    def _vg_characteristic_function(self, u, sigma, nu, theta, omega, S0):
        exponent = 1j * u * (np.log(S0) + (self.r + omega) * self.T)
        denominator = np.power(np.abs(1 - 1j * theta * nu * u + 0.5 * sigma**2 * nu * u**2), self.T / nu)
        return np.exp(exponent) / denominator



def replicate_table_2() -> List[Tuple[float, float]]:
    S0_values = [40, 50, 60, 70]
    weights = [1/3] * 3
    sigmas = [0.2, 0.4, 0.6]
    K_values = [50, 55, 60]
    T = 1
    r = 0.05
    rho = 0.5

    sigma_vg = 0.57
    nu_vg = 0.75
    theta_vg = -0.95

    results = []
    for K in K_values:
        assets = [AssetParameters(S0, weight, sigma) for S0, weight, sigma in zip(S0_values, weights, sigmas)]
        pricer = VGBasketOptionPricer(assets, K, T, r, sigma_vg, nu_vg, theta_vg, rho)
        price = pricer.price()
        results.append((K, price))

    return results

def replicate_table_3() -> List[Tuple[float, float, float]]:
    S0_values = [100, 100]
    weights = [0.5, 0.5]
    sigmas = [0.2, 0.4]  # Assuming these are the volatilities given for the first row
    K = 105.13
    T = 1
    r = 0.05
    rho = 0.5

    # todo: Ensure to explicitly mention that these are coming from research
    sigma_vg = 0.3477
    nu_vg = 0.4932
    theta_vg = -0.3919

    results = []
    for sigma in sigmas:
        assets = [AssetParameters(S0, weight, sigma) for S0, weight in zip(S0_values, weights)]
        pricer = VGBasketOptionPricer(assets, K, T, r, sigma_vg, nu_vg, theta_vg, rho)
        price = pricer.price()
        results.append((sigma, rho, price))

    return results

if __name__ == "__main__":
    print("Replicating Table 2:")
    table_2_results = replicate_table_2()
    for K, price in table_2_results:
        print(f"K = {K:.2f}, MM Price = {price:.4f}")

    print("\nReplicating Table 3:")
    table_3_results = replicate_table_3()
    for sigma, rho, price in table_3_results:
        print(f"sigma = {sigma:.1f}, rho = {rho:.1f}, MM Price = {price:.4f}")
