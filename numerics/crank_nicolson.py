import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.sparse import identity
from scipy.sparse.linalg import bicgstab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HestonModel:
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


class CrankNicolsonSolver:
    def __init__(self, model, Smax, vmax, M, N, L):
        self.model = model
        self.Smax = Smax
        self.vmax = vmax
        self.M = M
        self.N = N
        self.L = L

        self.dS = Smax / M
        self.dv = vmax / N
        self.dt = model.T / (L * 10)

        self.S = np.linspace(0, Smax, M + 1)
        self.v = np.linspace(0, vmax, N + 1)

        self.grid = np.zeros((M + 1, N + 1))

    def set_initial_condition(self):
        for i in range(self.M + 1):
            self.grid[i, :] = np.maximum(self.S[i] - self.model.K, 0)

    # def set_boundary_conditions(self, tau):
    #     # S = 0 boundary (Dirichlet)
    #     self.grid[0, :] = 0
    #
    #     # S = Smax boundary (Dirichlet for simplicity)
    #     self.grid[-1, :] = (self.Smax - self.model.K * np.exp(-self.model.r * tau))
    #
    #     # v = 0 boundary (simple Neumann condition)
    #     self.grid[1:-1, 0] = self.grid[1:-1, 1]
    #
    #     # v = vmax boundary (simple Neumann condition)
    #     self.grid[1:-1, -1] = self.grid[1:-1, -2]

    # def set_boundary_conditions(self, tau):
    #     # S = 0 boundary
    #     self.grid[0, :] = 0
    #     # S = Smax boundary
    #     self.grid[-1, :] = self.Smax - self.model.K * np.exp(-self.model.r * tau)
    #     # v = 0 boundary (Neumann condition)
    #     self.grid[1:-1, 0] = self.grid[1:-1, 1]
    #     # v = vmax boundary (Neumann condition)
    #     self.grid[1:-1, -1] = self.grid[1:-1, -2]

    def set_boundary_conditions(self, tau):
        # S = 0 boundary (Dirichlet)
        self.grid[0, :] = 0

        # S = Smax boundary (Neumann)
        self.grid[-1, :] = (self.Smax - self.model.K * np.exp(-self.model.r * tau))

        # Introduce a scaling factor to prevent large numbers
        scale_factor = 1e-3  # Scale down grid values
        scaled_grid = self.grid * scale_factor

        # v = 0 boundary using forward difference (equation 4.32)
        try:
            forward_diff = (-3 * scaled_grid[1:-1, 0] + 4 * scaled_grid[1:-1, 1] - scaled_grid[1:-1, 2]) / (2 * self.dv)
            forward_diff = np.clip(forward_diff, -1e10, 1e10)  # Clamping to prevent overflow
            self.grid[1:-1, 0] = forward_diff / scale_factor  # Rescale back after operation
        except OverflowError:
            print("Overflow detected during forward difference at v=0")

        # v = vmax boundary using backward difference (equation 4.33)
        try:
            backward_diff = (scaled_grid[1:-1, -3] - 4 * scaled_grid[1:-1, -2] + 3 * scaled_grid[1:-1, -1]) / (
                        2 * self.dv)
            backward_diff = np.clip(backward_diff, -1e10, 1e10)  # Clamping to prevent overflow
            self.grid[1:-1, -1] = backward_diff / scale_factor  # Rescale back after operation
        except OverflowError:
            print("Overflow detected during backward difference at v=vmax")


    def create_operators(self):
        M, N = self.M - 1, self.N - 1
        size = M * N

        # Define coefficients
        a = lambda i, j: 0.5 * self.v[j] * (i * self.dS) ** 2 / self.dS ** 2
        b = lambda j: 0.5 * self.model.sigma ** 2 * self.v[j] / self.dv ** 2
        c = lambda i, j: 0.25 * self.model.rho * self.model.sigma * i * self.dS * self.v[j] / (self.dS * self.dv)

        # Create diagonals
        diag = np.zeros(size)
        up1 = np.zeros(size - 1)
        up2 = np.zeros(size - N)
        left1 = np.zeros(size - 1)
        left2 = np.zeros(size - M)

        for j in range(N):
            for i in range(M):
                idx = j * M + i
                if idx < size:
                    diag[idx] = -(a(i + 1, j + 1) + b(j + 1) + 0.5 * self.model.r)
                if i < M - 1:
                    up1[idx] = 0.5 * a(i + 1, j + 1) - c(i + 1, j + 1)
                if j < N - 1:
                    up2[idx] = 0.5 * b(j + 1)
                if i > 0:
                    left1[idx - 1] = 0.5 * a(i + 1, j + 1) + c(i + 1, j + 1)
                if j > 0:
                    left2[idx - M] = 0.5 * b(j + 1)

        # Construct sparse matrices
        A = diags([left2, left1, diag, up1, up2], [-M, -1, 0, 1, M], shape=(size, size), format='csr')
        # Add a small regularization term to stabilize the matrix
        regularization = 1e-5 * identity(size, format='csr')
        A = A + regularization

        I = eye(size)

        # Debugging: Check condition number again after regularization
        print("Condition Number after Regularization:")
        print(np.linalg.cond(A.toarray()))

        return I - 0.5 * self.dt * A, I + 0.5 * self.dt * A

    def solve(self):
        self.set_initial_condition()
        A, B = self.create_operators()

        for l in range(1, self.L + 1):
            tau = (self.L - l) * self.dt
            self.set_boundary_conditions(tau)

            # Prepare right-hand side
            rhs = self.grid[1:-1, 1:-1].flatten()
            rhs = B @ rhs

            # print(f"RHS at time step {l}: {rhs}")

            # Solve linear system
            x = spsolve(A, rhs)

            # print(f"Solution vector at time step {l}: {x}")
            self.grid[1:-1, 1:-1] = x.reshape((self.M - 1, self.N - 1))

        # Interpolate to find the price at S0, v0 (return single value)
        i = int(self.model.S0 / self.dS)
        j = int(self.model.v0 / self.dv)
        w1 = (self.model.S0 - self.S[i]) / self.dS
        w2 = (self.model.v0 - self.v[j]) / self.dv
        price = (1 - w1) * (1 - w2) * self.grid[i, j] + \
                w1 * (1 - w2) * self.grid[i + 1, j] + \
                (1 - w1) * w2 * self.grid[i, j + 1] + \
                w1 * w2 * self.grid[i + 1, j + 1]

        return price, self.grid # Return both the single price and the entire grid


def main():
    # Model parameters based on the paper
    S0, K, r, T = 100, 100, 0.03, 1
    kappa, theta, sigma, rho, v0 = 2, 0.2, 0.3, 0.8, 0.2

    # Numerical parameters based on the paper
    Smax, vmax = 200, 1
    M, N, L = 40, 20, 4000

    model = HestonModel(S0, K, r, T, kappa, theta, sigma, rho, v0)
    solver = CrankNicolsonSolver(model, Smax, vmax, M, N, L)

    # Solve for the option price and retrieve the entire grid
    option_price, option_price_grid = solver.solve()
    print(f"European Call Option Price: {option_price:.4f}")

    # # Creating the mesh grid for S and V
    S = np.linspace(0, Smax, M + 1)
    V = np.linspace(0, vmax, N + 1)
    S, V = np.meshgrid(S, V)

    # Transpose the option_price_grid to match S and V grids
    option_price_grid = option_price_grid.T

    # Finite difference parameters
    dS = Smax / M
    dv = vmax / N

    # Implementing Delta using finite difference (eq. 4.26b)
    Delta = (option_price_grid[2:, 1:-1] - option_price_grid[:-2, 1:-1]) / (2 * dS)

    # Implementing Gamma using finite difference (eq. 4.26d)
    Gamma = (option_price_grid[2:, 1:-1] - 2 * option_price_grid[1:-1, 1:-1] + option_price_grid[:-2, 1:-1]) / (dS ** 2)

    # Adjust S and V for Delta and Gamma plots (since they are reduced by 1 in both axes)
    S_reduced = S[1:-1, 1:-1]
    V_reduced = V[1:-1, 1:-1]

    # Plot results: Price, Delta, Gamma
    fig = plt.figure(figsize=(18, 12))

    # Option Price Surface Plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(S, V, option_price_grid, cmap='viridis')
    ax1.set_title('Option Price Surface')
    ax1.set_xlabel('S')
    ax1.set_ylabel('V')
    ax1.set_zlabel('Price')

    # Delta Plot
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(S_reduced, V_reduced, Delta, cmap='viridis')
    ax2.set_title('Delta (∂Price/∂S)')
    ax2.set_xlabel('S')
    ax2.set_ylabel('V')
    ax2.set_zlabel('Delta')

    # Gamma Plot
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(S_reduced, V_reduced, Gamma, cmap='viridis')
    ax3.set_title('Gamma (∂²Price/∂S²)')
    ax3.set_xlabel('S')
    ax3.set_ylabel('V')
    ax3.set_zlabel('Gamma')

    plt.show()

if __name__ == "__main__":
    main()
