import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

import FNO
import SNO


# ===========================
# CONFIG
# ===========================
PDE = "Wave Equation"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Interval: x in I
I = [0.0, 1.0]
c = 0.05
NX = 100
x = np.linspace(I[0], I[1], NX)
dx = x[1] - x[0]

# NOTE: these t, dt are not actually used for stepping (solve_ivp_fd uses dt=0.001 by default)
TI = [0.0, 1.0]
NT = 1000
t = np.linspace(TI[0], TI[1], NT)
dt = t[1] - t[0]

MAX_CHEBYSHEV_DEGREE = 20

FRAMES = 3000
INTERVAL_MS = 50
FPS = 200


# ===========================
# RANDOM INITIAL CONDITION
# ===========================
def random_func_T(x_array):
    x_mapped = (2.0 * (x_array - I[0]) / (I[1] - I[0])) - 1.0
    cheb_coeffs = np.random.uniform(-1.0, 1.0, size=MAX_CHEBYSHEV_DEGREE + 1)
    cheb_polys = [np.polynomial.chebyshev.Chebyshev.basis(i)(x_mapped) for i in range(len(cheb_coeffs))]
    func = sum(cc * p * (x_array - I[0]) * (I[1] - x_array) for cc, p in zip(cheb_coeffs, cheb_polys))
    func /= np.max(np.abs(func))
    return func


# ===========================
# LOAD TRAINED MODELS
# ===========================
# --- FNO
model_2 = FNO.get_fno_model()
state_dict_2 = torch.load("checkpoints/FNO_epoch_500.pth", map_location=device)
state_dict_2 = {k.replace("module.", ""): v for k, v in state_dict_2.items()}
model_2.load_state_dict(state_dict_2, strict=True)
model_2.to(device)
model_2.eval()
LABEL2 = "FNO"

# --- SNO (your SNO.py self-adjoint symplectic model)
model_3 = SNO.get_model()  # uses config inside SNO.py
state_dict_3 = torch.load("checkpoints/SNO_epoch_500.pth", map_location=device)  # <-- change epoch
if isinstance(state_dict_3, dict) and "_metadata" in state_dict_3:
    del state_dict_3["_metadata"]
state_dict_3 = {k.replace("module.", ""): v for k, v in state_dict_3.items()}
model_3.load_state_dict(state_dict_3, strict=True)
model_3.to(device)
model_3.eval()
LABEL3 = "SNO(Self-Adjoint)"


# ===========================
# FD SOLVER (ONE STEP)
# ===========================
def solve_ivp_fd(f, g, dt_step=0.001):
    nx_local = len(f)

    def wave_eq(_t, y):
        u = y[:nx_local]
        v = y[nx_local:]
        dudt = v
        dvdt = np.zeros(nx_local)
        dvdt[1:-1] = (c / dx) ** 2 * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        return np.concatenate([dudt, dvdt])

    y0 = np.concatenate([f, g])
    sol = solve_ivp(wave_eq, [0.0, dt_step], y0, method="RK45", rtol=1e-8, atol=1e-8)
    f_new = sol.y[:nx_local, -1].T
    g_new = sol.y[nx_local:, -1].T
    return f_new, g_new


# ===========================
# MODEL ONE-STEP WRAPPERS
# ===========================
def new_2(f_arr, g_arr):
    inp = torch.tensor(np.stack([f_arr, g_arr], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model_2(inp).detach().cpu().squeeze().numpy()  # expected (2, NX)
    return pred[0], pred[1]


def new_3(f_arr, g_arr):
    inp = torch.tensor(np.stack([f_arr, g_arr], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model_3(inp).detach().cpu().squeeze().numpy()  # expected (2, NX)
    return pred[0], pred[1]


# ===========================
# ENERGY
# ===========================
def calculate_energy(f_arr, g_arr, x_arr):
    df_dx = np.gradient(f_arr, x_arr)
    return 0.5 * np.sum(g_arr**2 + c**2 * df_dx**2) * (x_arr[1] - x_arr[0])


# ===========================
# INITIAL CONDITIONS
# ===========================
f0 = random_func_T(x) / 3.0
g0 = random_func_T(x) / 3.0

f_1, g_1 = f0.copy(), g0.copy()  # FD
f_2, g_2 = f0.copy(), g0.copy()  # FNO
f_3, g_3 = f0.copy(), g0.copy()  # SNO

energy_1 = [calculate_energy(f_1, g_1, x)]
energy_2 = [calculate_energy(f_2, g_2, x)]
energy_3 = [calculate_energy(f_3, g_3, x)]


# ===========================
# PLOT SETUP
# ===========================
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle(f"{PDE}: FD vs {LABEL2} vs {LABEL3}", fontsize=18)

ax1.set_title("Displacement")
ax2.set_title("Velocity")
ax3.set_title(f"Energy Evolution of {PDE}")

lines = {
    "f0": ax1.plot(x, f0, label="Initial Displacement", color="black")[0],
    "f_1": ax1.plot(x, f_1, label="FD", color="blue")[0],
    "f_2": ax1.plot(x, f_2, label=LABEL2, color="green")[0],
    "f_3": ax1.plot(x, f_3, label=LABEL3, color="red")[0],

    "g0": ax2.plot(x, g0, label="Initial Velocity", color="black")[0],
    "g_1": ax2.plot(x, g_1, label="FD", color="blue")[0],
    "g_2": ax2.plot(x, g_2, label=LABEL2, color="green")[0],
    "g_3": ax2.plot(x, g_3, label=LABEL3, color="red")[0],

    "energy_1": ax3.plot([], [], label="FD", color="blue")[0],
    "energy_2": ax3.plot([], [], label=LABEL2, color="green")[0],
    "energy_3": ax3.plot([], [], label=LABEL3, color="red")[0],
}

for ax in [ax1, ax2]:
    ax.set_xlim(I[0], I[1])
    ax.set_ylim(-2.0, 2.0)
    ax.legend()
    ax.grid(True)

ax3.set_xlim(0, FRAMES)
ax3.set_ylim(0.0, 0.3)
ax3.legend()
ax3.grid(True)


# ===========================
# ANIMATION UPDATE
# ===========================
def update(_frame):
    global f_1, g_1, f_2, g_2, f_3, g_3

    # One step for each method
    f_1, g_1 = solve_ivp_fd(f_1, g_1, dt_step=0.001)
    f_2, g_2 = new_2(f_2, g_2)
    f_3, g_3 = new_3(f_3, g_3)

    # Energies
    energy_1.append(calculate_energy(f_1, g_1, x))
    energy_2.append(calculate_energy(f_2, g_2, x))
    energy_3.append(calculate_energy(f_3, g_3, x))

    # Update curves
    lines["f_1"].set_ydata(f_1)
    lines["f_2"].set_ydata(f_2)
    lines["f_3"].set_ydata(f_3)

    lines["g_1"].set_ydata(g_1)
    lines["g_2"].set_ydata(g_2)
    lines["g_3"].set_ydata(g_3)

    lines["energy_1"].set_data(range(len(energy_1)), energy_1)
    lines["energy_2"].set_data(range(len(energy_2)), energy_2)
    lines["energy_3"].set_data(range(len(energy_3)), energy_3)

    return tuple(lines.values())


ani = FuncAnimation(fig, update, frames=FRAMES, interval=INTERVAL_MS, blit=False)

SAVE_PATH = f"{PDE}_comparison_fd_{LABEL2}_{LABEL3}.mp4"
ani.save(SAVE_PATH, writer="ffmpeg", fps=FPS)
print("Saved:", SAVE_PATH)

plt.show()
