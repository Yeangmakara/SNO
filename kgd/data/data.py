import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# Interval
I = [0, 1]
NX = 100 * (I[1] - I[0])
x = np.linspace(I[0], I[1], int(NX))
dx = x[1] - x[0]
MAX_CHEBYSHEV_DEGREE = 20

# Chebyshev-random initial function with Dirichlet boundary
def random_func_T(x):
    x_mapped = 2 * (x - I[0]) / (I[1] - I[0]) - 1
    coeffs = np.random.uniform(-1, 1, MAX_CHEBYSHEV_DEGREE + 1)
    cheb_polys = [np.polynomial.chebyshev.Chebyshev.basis(i)(x_mapped) for i in range(len(coeffs))]
    f = sum(c * p * (x - I[0]) * (I[1] - x) for c, p in zip(coeffs, cheb_polys))
    return f / np.max(np.abs(f))

# Laplacian matrix with Dirichlet BCs
def laplacian_matrix(N, dx):
    D2 = -2 * np.eye(N)
    for i in range(1, N - 1):
        D2[i, i - 1] = D2[i, i + 1] = 1
    D2[0, :] = D2[-1, :] = 0
    return D2 / dx**2

D2 = laplacian_matrix(int(NX), dx)

# Solve nonlinear Klein-Gordon u_tt = c^2*u_xx + u + u^3
def solve_ivp_fd(f, g, dt=0.001, c=0.1):
    N = len(f)
    def kg_eq(t, y):
        q, p = y[:N], y[N:]
        dq = p
        dp = c**2 * (D2 @ q) + q + q**3
        dp[0] = dp[-1] = 0  # Dirichlet BCs
        return np.concatenate([dq, dp])
    y0 = np.concatenate([f, g])
    sol = solve_ivp(kg_eq, [0, dt], y0, rtol=1e-8, atol=1e-8)
    return sol.y[:N, -1], sol.y[N:, -1]


# Energy functional for nonlinear Klein-Gordon
def calculate_energy(f, g, x, c=0.1):
    dq = np.gradient(f, dx)
    energy_density = 0.5 * g**2 + 0.5 * c**2 * dq**2 + 0.5 *f**2 + 0.25 * f**4
    return np.sum(energy_density) * dx


def generate_dataset(num_samples,dt=0.001):
    dataset = []
    for sam in range(num_samples):
        if sam%1000==999:
            print(sam+1)
        f = random_func_T(x)  # Random initial displacement
        g = random_func_T(x)  # Random initial velocity

        f_new, g_new = solve_ivp_fd(f, g,dt)
        input = np.concatenate([f, g])  # current state
        target = [f_new, g_new]  # Next state at x[j]
        dataset.append((input, target))
    return dataset

def save_dataset(dataset, filename="data.npz"):
    inputs = np.array([data[0] for data in dataset])
    targets = np.array([data[1] for data in dataset])

    np.savez(filename, inputs=inputs, targets=targets)



if __name__=="__main__":
    # Generate and save data
    num_samples = 100000
    file_name="data.npz"
    dataset = generate_dataset(num_samples)
    print("ok")
    save_dataset(dataset,file_name)
    print(f"Dataset saved to {file_name} with {num_samples} samples.")




"""





# Initial data
f0 = random_func_T(x) / 2
g0 = random_func_T(x) / 2
f_fd, g_fd = f0.copy(), g0.copy()
energy_fd = [calculate_energy(f_fd, g_fd, x)]

# Plot setup
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 6))
lines = {
    "f0": ax1.plot(x, f0, label="Initial Displacement", color="black")[0],
    "f_fd": ax1.plot(x, f_fd, label="Displacement FD", color="blue")[0],
    "g0": ax2.plot(x, g0, label="Initial Velocity", color="black")[0],
    "g_fd": ax2.plot(x, g_fd, label="Velocity FD", color="blue")[0],
    "energy_fd": ax3.plot([], [], label="Energy", color="blue")[0]
}
for ax in [ax1, ax2]:
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid()
ax3.set_xlim(0, 1000)
ax3.set_ylim(0, energy_fd[0]+1)
ax3.legend()
ax3.grid()

# Update function
def update(frame):
    global f_fd, g_fd
    f_fd, g_fd = solve_ivp_fd(f_fd, g_fd, dt=0.001)
    energy_fd.append(calculate_energy(f_fd, g_fd, x))
    lines["f_fd"].set_ydata(f_fd)
    lines["g_fd"].set_ydata(g_fd)
    lines["energy_fd"].set_data(np.arange(len(energy_fd)), energy_fd)
    return list(lines.values())

# Run animation
ani = FuncAnimation(fig, update, frames=1000, interval=1)
plt.tight_layout()
plt.show()



"""
