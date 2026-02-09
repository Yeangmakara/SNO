import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

import FNO
import 


# ===========================
# CONFIG
# ===========================
PDE = "Wave Equation"

# Spatial domain
I = [0.0, 1.0]
c = 0.05
NX = 100
x = np.linspace(I[0], I[1], NX)
dx = x[1] - x[0]

# Time stepping for rollout (one-step model)
DT = 0.001
NUM_STEPS = 3000   # final time = NUM_STEPS * DT

# Random initial condition
MAX_CHEBYSHEV_DEGREE = 20
SEED = 0

# Checkpoints (EDIT THESE)
CHECKPOINT_FNO = "checkpoints/FNO_epoch_500.pth"
CHECKPOINT_SNO = "checkpoints/SNO_epoch_500.pth" 

# Output
SAVE_DIR = "figures_fno_vs_sno"
SAVE_DPI = 300
FIELD_CMAP = "RdBu_r"   # diverging cmap is nice for signed waves
ERROR_CMAP = "RdBu_r"

SNAPSHOT_TIMES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ===========================
# DEVICE
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================
# UTILITIES
# ===========================
def safe_name(s: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in s)

def random_func_T(x_array):
    x_mapped = (2.0 * (x_array - I[0]) / (I[1] - I[0])) - 1.0
    cheb_coeffs = np.random.uniform(-1.0, 1.0, size=MAX_CHEBYSHEV_DEGREE + 1)
    cheb_polys = [np.polynomial.chebyshev.Chebyshev.basis(i)(x_mapped) for i in range(len(cheb_coeffs))]
    func = sum(cc * p * (x_array - I[0]) * (I[1] - x_array) for cc, p in zip(cheb_coeffs, cheb_polys))
    func /= np.max(np.abs(func))
    return func

def solve_ivp_fd_one_step(f_now, g_now, dt_step=0.001):
    nx_local = len(f_now)

    def wave_eq(_t, y):
        u = y[:nx_local]
        v = y[nx_local:]
        dudt = v
        dvdt = np.zeros(nx_local)
        dvdt[1:-1] = (c / dx) ** 2 * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        return np.concatenate([dudt, dvdt])

    y0 = np.concatenate([f_now, g_now])
    sol = solve_ivp(wave_eq, [0.0, dt_step], y0, method="RK45", rtol=1e-8, atol=1e-8)
    y_end = sol.y[:, -1]
    f_new = y_end[:nx_local]
    g_new = y_end[nx_local:]
    return f_new, g_new

def model_step(model, f_arr, g_arr):
    inp = torch.tensor(np.stack([f_arr, g_arr], axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).detach().cpu().squeeze().numpy()  # (2, NX)
    return pred[0], pred[1]

def calculate_energy(f_arr, g_arr, x_arr):
    df_dx = np.gradient(f_arr, x_arr)
    return 0.5 * np.sum(g_arr**2 + c**2 * df_dx**2) * (x_arr[1] - x_arr[0])

def symm_limits(*arrays):
    max_abs = 0.0
    for a in arrays:
        max_abs = max(max_abs, float(np.max(np.abs(a))))
    if max_abs == 0.0:
        max_abs = 1.0
    return -max_abs, max_abs


# ===========================
# LOAD MODELS
# ===========================
# FNO
model_fno = fno.get_fno_model()
sd_fno = torch.load(CHECKPOINT_FNO, map_location=device)
sd_fno = {k.replace("module.", ""): v for k, v in sd_fno.items()}
model_fno.load_state_dict(sd_fno, strict=True)
model_fno.to(device)
model_fno.eval()
LABEL_FNO = "FNO"

# SNO (your self-adjoint symplectic model in SNO.py)
model_sno = SNO.get_model()
sd_sno = torch.load(CHECKPOINT_SNO, map_location=device)
if isinstance(sd_sno, dict) and "_metadata" in sd_sno:
    del sd_sno["_metadata"]
sd_sno = {k.replace("module.", ""): v for k, v in sd_sno.items()}
model_sno.load_state_dict(sd_sno, strict=True)
model_sno.to(device)
model_sno.eval()
LABEL_SNO = "SNO"


# ===========================
# INITIAL CONDITIONS
# ===========================
f0 = random_func_T(x) / 3.0
g0 = random_func_T(x) / 3.0

time = np.arange(NUM_STEPS + 1) * DT


# ===========================
# ROLLOUT STORAGE
# ===========================
f_fd  = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)
g_fd  = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)
f_fno = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)
g_fno = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)
f_sno = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)
g_sno = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)

energy_fd  = np.zeros(NUM_STEPS + 1, dtype=np.float64)
energy_fno = np.zeros(NUM_STEPS + 1, dtype=np.float64)
energy_sno = np.zeros(NUM_STEPS + 1, dtype=np.float64)

# t=0
f_fd[0],  g_fd[0]  = f0.copy(), g0.copy()
f_fno[0], g_fno[0] = f0.copy(), g0.copy()
f_sno[0], g_sno[0] = f0.copy(), g0.copy()

energy_fd[0]  = calculate_energy(f_fd[0],  g_fd[0],  x)
energy_fno[0] = calculate_energy(f_fno[0], g_fno[0], x)
energy_sno[0] = calculate_energy(f_sno[0], g_sno[0], x)


# ===========================
# ROLLOUT LOOP (NO VIDEO)
# ===========================
print(f"Rolling out {NUM_STEPS} steps with dt={DT} ...")
for n in range(1, NUM_STEPS + 1):
    f_fd[n],  g_fd[n]  = solve_ivp_fd_one_step(f_fd[n - 1],  g_fd[n - 1],  dt_step=DT)
    f_fno[n], g_fno[n] = model_step(model_fno, f_fno[n - 1], g_fno[n - 1])
    f_sno[n], g_sno[n] = model_step(model_sno, f_sno[n - 1], g_sno[n - 1])

    energy_fd[n]  = calculate_energy(f_fd[n],  g_fd[n],  x)
    energy_fno[n] = calculate_energy(f_fno[n], g_fno[n], x)
    energy_sno[n] = calculate_energy(f_sno[n], g_sno[n], x)

print("Rollout done.")


# ===========================
# FIGURE HELPERS
# ===========================
def save_spacetime_3col(field_fd_arr, field_fno_arr, field_sno_arr, title, filename_prefix, cmap):
    vmin, vmax = symm_limits(field_fd_arr, field_fno_arr, field_sno_arr)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    extent = [x[0], x[-1], time[0], time[-1]]

    im0 = axes[0].imshow(field_fd_arr, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title("FD")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    im1 = axes[1].imshow(field_fno_arr, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(LABEL_FNO)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")

    im2 = axes[2].imshow(field_sno_arr, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[2].set_title(LABEL_SNO)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")

    cbar = fig.colorbar(im2, ax=axes, shrink=0.95)
    cbar.set_label(title)

    fig.suptitle(f"{PDE}: {title} (space–time)", y=1.05)

    #pdf_path = os.path.join(SAVE_DIR, f"{filename_prefix}.pdf")
    png_path = os.path.join(SAVE_DIR, f"{filename_prefix}.png")
    #fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

def save_spacetime_error_2col(err_fno, err_sno, title, filename_prefix, cmap):
    vmin, vmax = symm_limits(err_fno, err_sno)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    extent = [x[0], x[-1], time[0], time[-1]]

    im0 = axes[0].imshow(err_fno, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title(f"{LABEL_FNO} - FD")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    im1 = axes[1].imshow(err_sno, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title(f"{LABEL_SNO} - FD")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")

    cbar = fig.colorbar(im1, ax=axes, shrink=0.95)
    cbar.set_label(title)

    fig.suptitle(f"{PDE}: {title} error (space–time)", y=1.05)

    #pdf_path = os.path.join(SAVE_DIR, f"{filename_prefix}.pdf")
    png_path = os.path.join(SAVE_DIR, f"{filename_prefix}.png")
    #fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

def save_energy_plot():
    fig = plt.figure(figsize=(7, 4), constrained_layout=True)
    plt.plot(time, energy_fd, label="FD")
    plt.plot(time, energy_fno, label=LABEL_FNO)
    plt.plot(time, energy_sno, label=LABEL_SNO)
    plt.xlabel("t")
    plt.ylabel("Energy")
    plt.title(f"{PDE}: Energy evolution")
    plt.grid(True, alpha=0.3)
    plt.legend()

    #pdf_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_energy.pdf")
    png_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_energy.png")
    #fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

def save_relative_error_curves():
    eps = 1e-12

    fd_norm_f = np.linalg.norm(f_fd, axis=1) + eps
    fd_norm_g = np.linalg.norm(g_fd, axis=1) + eps

    rel_f_fno = np.linalg.norm(f_fno - f_fd, axis=1) / fd_norm_f
    rel_f_sno = np.linalg.norm(f_sno - f_fd, axis=1) / fd_norm_f

    rel_g_fno = np.linalg.norm(g_fno - g_fd, axis=1) / fd_norm_g
    rel_g_sno = np.linalg.norm(g_sno - g_fd, axis=1) / fd_norm_g

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    axes[0].plot(time, rel_f_fno, label=f"{LABEL_FNO} vs FD")
    axes[0].plot(time, rel_f_sno, label=f"{LABEL_SNO} vs FD")
    axes[0].set_title("Relative L2 error (displacement)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("||f_model - f_FD|| / ||f_FD||")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(time, rel_g_fno, label=f"{LABEL_FNO} vs FD")
    axes[1].plot(time, rel_g_sno, label=f"{LABEL_SNO} vs FD")
    axes[1].set_title("Relative L2 error (velocity)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("||g_model - g_FD|| / ||g_FD||")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{PDE}: Relative errors vs FD")

    #pdf_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_relative_errors.pdf")
    png_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_relative_errors.png")
    #fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)

def save_snapshots():
    snapshot_indices = []
    for ts in SNAPSHOT_TIMES:
        idx = int(np.argmin(np.abs(time - ts)))
        snapshot_indices.append(idx)

    fig, axes = plt.subplots(2, len(snapshot_indices), figsize=(4.0 * len(snapshot_indices), 6.0), constrained_layout=True)

    for j, idx in enumerate(snapshot_indices):
        axes[0, j].plot(x, f_fd[idx], label="FD")
        axes[0, j].plot(x, f_fno[idx], label=LABEL_FNO)
        axes[0, j].plot(x, f_sno[idx], label=LABEL_SNO)
        axes[0, j].set_title(f"f(x,t) at t≈{time[idx]:.3f}")
        axes[0, j].set_xlabel("x")
        axes[0, j].grid(True, alpha=0.3)

        axes[1, j].plot(x, g_fd[idx], label="FD")
        axes[1, j].plot(x, g_fno[idx], label=LABEL_FNO)
        axes[1, j].plot(x, g_sno[idx], label=LABEL_SNO)
        axes[1, j].set_title(f"g(x,t) at t≈{time[idx]:.3f}")
        axes[1, j].set_xlabel("x")
        axes[1, j].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Displacement")
    axes[1, 0].set_ylabel("Velocity")
    axes[0, -1].legend(loc="upper right")

    fig.suptitle(f"{PDE}: Snapshots")

    #pdf_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_snapshots.pdf")
    png_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_snapshots.png")
    #fig.savefig(pdf_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)


# ===========================
# MAKE FIGURES
# ===========================
name = safe_name(PDE)

save_spacetime_3col(f_fd, f_fno, f_sno, "Displacement f", f"{name}_spacetime_displacement", FIELD_CMAP)
save_spacetime_3col(g_fd, g_fno, g_sno, "Velocity g",     f"{name}_spacetime_velocity",     FIELD_CMAP)

# Error fields
err_f_fno = f_fno - f_fd
err_f_sno = f_sno - f_fd
err_g_fno = g_fno - g_fd
err_g_sno = g_sno - g_fd

save_spacetime_error_2col(err_f_fno, err_f_sno, "Displacement error", f"{name}_spacetime_error_displacement", ERROR_CMAP)
save_spacetime_error_2col(err_g_fno, err_g_sno, "Velocity error",     f"{name}_spacetime_error_velocity",     ERROR_CMAP)

save_energy_plot()
save_relative_error_curves()
save_snapshots()

# Optional: save raw arrays for later plotting / paper reproducibility
np.savez(
    os.path.join(SAVE_DIR, f"{name}_rollout_data.npz"),
    x=x,
    time=time,
    f_fd=f_fd, g_fd=g_fd,
    f_fno=f_fno, g_fno=g_fno,
    f_sno=f_sno, g_sno=g_sno,
    energy_fd=energy_fd, energy_fno=energy_fno, energy_sno=energy_sno
)

print("Saved paper figures to:", SAVE_DIR)
