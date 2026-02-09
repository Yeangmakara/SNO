import os
import numpy as np
import torch
import matplotlib.pyplot as matplotlib_pyplot
from scipy.integrate import solve_ivp

import FNO as model1
import SNO as model2
import GNO as model3
import CNO as model4


# ===========================
# USER-EDITABLE SETTINGS
# ===========================
PDE = "Electromagnetic Wave (Maxwell 1D)"

# Spatial domain
I = [0.0, 1.0]
C_EM = 1.0
NX = 100
X = np.linspace(I[0], I[1], NX)
DX = X[1] - X[0]

# Rollout
DT = 0.001
NUM_STEPS = 500

# Random initial condition
MAX_CHEBYSHEV_DEGREE = 10
SEED = 0

# Output
SAVE_DIR = "figures4"
SAVE_DPI = 300
FIELD_CMAP = "RdBu_r"
ERROR_CMAP = "RdBu_r"
SNAPSHOT_TIMES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# ---------------------------
# Labels (used in plots)
# ---------------------------
LABEL_FNO = "FNO"
LABEL_SNO = "SNO (ours)"
LABEL_GNO = "GNO"
LABEL_CNO = "CNO"

# ---------------------------
# Checkpoints
# ---------------------------
CHECKPOINT_FNO = "checkpoints/FNO_epoch_500.pth"
CHECKPOINT_SNO = "checkpoints/SNO_epoch_500.pth"
CHECKPOINT_GNO = "checkpoints/GNO_epoch_500.pth"
CHECKPOINT_CNO = "checkpoints/CNO_epoch_500.pth"

# ---- GNO hyperparameters (must match training) ----
GNO_IN_CHANNELS = 2
GNO_OUT_CHANNELS = 2
GNO_HIDDEN_CHANNELS = 256
GNO_NUM_LAYERS = 7
GNO_NEIGHBOR_RADIUS = 8
GNO_KERNEL_HIDDEN_DIMS = [256, 256, 256]
GNO_KERNEL_ACTIVATION = "gelu"
GNO_AGGREGATION = "mean"
GNO_PERIODIC = False
GNO_X_MIN = I[0]
GNO_X_MAX = I[1]

# ---- CNO hyperparameters (must match training) ----
CNO_IN_CHANNELS = 2
CNO_OUT_CHANNELS = 2
CNO_WIDTH = 128
CNO_DEPTH = 5
CNO_KERNEL_SIZE = 5
CNO_USE_CIRCULAR_PADDING = False


# ===========================
# DEVICE / SEED
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

np.random.seed(SEED)
torch.manual_seed(SEED)
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================
# UTILITIES
# ===========================
def safe_name(S: str) -> str:
    return "".join(CH if (CH.isalnum() or CH in "-_") else "_" for CH in S)

def strip_module_prefix(STATE_DICT):
    if not isinstance(STATE_DICT, dict):
        return STATE_DICT
    if "_metadata" in STATE_DICT:
        del STATE_DICT["_metadata"]
    return {K.replace("module.", ""): V for K, V in STATE_DICT.items()}

def random_func_T(X_ARRAY):
    x_mapped = (2.0 * (X_ARRAY - I[0]) / (I[1] - I[0])) - 1.0
    cheb_coeffs = np.random.uniform(-1.0, 1.0, size=MAX_CHEBYSHEV_DEGREE + 1)
    cheb_polys = [np.polynomial.chebyshev.Chebyshev.basis(i)(x_mapped) for i in range(len(cheb_coeffs))]
    func = sum(CC * P * (X_ARRAY - I[0]) * (I[1] - X_ARRAY) for CC, P in zip(cheb_coeffs, cheb_polys))
    func /= np.max(np.abs(func)) + 1e-12
    return func

def solve_ivp_fd_one_step(E_NOW, B_NOW, DT_STEP):
    """
    Maxwell 1D:
        E_t = c * B_x
        B_t = c * E_x
    Central difference on interior, like em.py.
    """
    nx_local = len(E_NOW)

    def maxwell_eq(_t, y):
        E = y[:nx_local]
        B = y[nx_local:]

        dEdt = np.zeros(nx_local)
        dBdt = np.zeros(nx_local)

        dEdt[1:-1] = (C_EM / DX) * (B[2:] - B[:-2]) / 2.0
        dBdt[1:-1] = (C_EM / DX) * (E[2:] - E[:-2]) / 2.0

        return np.concatenate([dEdt, dBdt])

    y0 = np.concatenate([E_NOW, B_NOW])
    sol = solve_ivp(maxwell_eq, [0.0, DT_STEP], y0, method="RK45", rtol=1e-8, atol=1e-8)
    y_end = sol.y[:, -1]
    E_new = y_end[:nx_local]
    B_new = y_end[nx_local:]
    return E_new, B_new

def model_step(MODEL, E_ARR, B_ARR):
    inp = torch.tensor(np.stack([E_ARR, B_ARR], axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = MODEL(inp).detach().cpu().squeeze().numpy()  # (2, NX)
    return pred[0], pred[1]

def calculate_energy(E_ARR, B_ARR, X_ARR):
    dx_local = float(X_ARR[1] - X_ARR[0])
    energy = 0.5 * C_EM * float(np.sum((B_ARR ** 2) + (E_ARR ** 2)) * dx_local)
    return energy

def symm_limits(*arrays):
    max_abs = 0.0
    for a in arrays:
        max_abs = max(max_abs, float(np.max(np.abs(a))))
    if max_abs == 0.0:
        max_abs = 1.0
    return -max_abs, max_abs


# ===========================
# BUILD + LOAD MODELS
# ===========================
# FNO
MODEL_FNO = model1.get_fno_model()
SD_FNO = strip_module_prefix(torch.load(CHECKPOINT_FNO, map_location=DEVICE))
MODEL_FNO.load_state_dict(SD_FNO, strict=True)
MODEL_FNO.to(DEVICE).eval()

# SNO (ours)
MODEL_SNO = model2.get_model()
SD_SNO = strip_module_prefix(torch.load(CHECKPOINT_SNO, map_location=DEVICE))
MODEL_SNO.load_state_dict(SD_SNO, strict=True)
MODEL_SNO.to(DEVICE).eval()

# GNO
MODEL_GNO = model3.get_gno_model(
    NX=NX,
    IN_CHANNELS=GNO_IN_CHANNELS,
    OUT_CHANNELS=GNO_OUT_CHANNELS,
    HIDDEN_CHANNELS=GNO_HIDDEN_CHANNELS,
    NUM_LAYERS=GNO_NUM_LAYERS,
    NEIGHBOR_RADIUS=GNO_NEIGHBOR_RADIUS,
    KERNEL_HIDDEN_DIMS=GNO_KERNEL_HIDDEN_DIMS,
    KERNEL_ACTIVATION=GNO_KERNEL_ACTIVATION,
    AGGREGATION=GNO_AGGREGATION,
    X_MIN=GNO_X_MIN,
    X_MAX=GNO_X_MAX,
    PERIODIC=GNO_PERIODIC,
)
SD_GNO = strip_module_prefix(torch.load(CHECKPOINT_GNO, map_location=DEVICE))
MODEL_GNO.load_state_dict(SD_GNO, strict=True)
MODEL_GNO.to(DEVICE).eval()

# CNO
MODEL_CNO = model4.get_cno_model(
    IN_CHANNELS=CNO_IN_CHANNELS,
    OUT_CHANNELS=CNO_OUT_CHANNELS,
    WIDTH=CNO_WIDTH,
    DEPTH=CNO_DEPTH,
    KERNEL_SIZE=CNO_KERNEL_SIZE,
    USE_CIRCULAR_PADDING=CNO_USE_CIRCULAR_PADDING,
)
SD_CNO = strip_module_prefix(torch.load(CHECKPOINT_CNO, map_location=DEVICE))
MODEL_CNO.load_state_dict(SD_CNO, strict=True)
MODEL_CNO.to(DEVICE).eval()

MODELS = [
    {"label": LABEL_SNO, "model": MODEL_SNO},
    {"label": LABEL_FNO, "model": MODEL_FNO},
    {"label": LABEL_GNO, "model": MODEL_GNO},
    {"label": LABEL_CNO, "model": MODEL_CNO},
]


# ===========================
# INITIAL CONDITIONS
# ===========================
E0 = random_func_T(X) / 3.0
B0 = random_func_T(X) / 3.0
TIME = np.arange(NUM_STEPS + 1) * DT


# ===========================
# ROLLOUT STORAGE
# ===========================
E_FD = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)
B_FD = np.zeros((NUM_STEPS + 1, NX), dtype=np.float64)

E_MODELS = [np.zeros((NUM_STEPS + 1, NX), dtype=np.float64) for _ in MODELS]
B_MODELS = [np.zeros((NUM_STEPS + 1, NX), dtype=np.float64) for _ in MODELS]

ENERGY_FD = np.zeros(NUM_STEPS + 1, dtype=np.float64)
ENERGY_MODELS = [np.zeros(NUM_STEPS + 1, dtype=np.float64) for _ in MODELS]

# t=0
E_FD[0], B_FD[0] = E0.copy(), B0.copy()
for i in range(len(MODELS)):
    E_MODELS[i][0], B_MODELS[i][0] = E0.copy(), B0.copy()

ENERGY_FD[0] = calculate_energy(E_FD[0], B_FD[0], X)
for i in range(len(MODELS)):
    ENERGY_MODELS[i][0] = calculate_energy(E_MODELS[i][0], B_MODELS[i][0], X)


# ===========================
# ROLLOUT LOOP
# ===========================
print(f"Rolling out {NUM_STEPS} steps with dt={DT} ...")
for n in range(1, NUM_STEPS + 1):
    E_FD[n], B_FD[n] = solve_ivp_fd_one_step(E_FD[n - 1], B_FD[n - 1], DT_STEP=DT)

    for i, entry in enumerate(MODELS):
        e_next, b_next = model_step(entry["model"], E_MODELS[i][n - 1], B_MODELS[i][n - 1])
        E_MODELS[i][n] = e_next
        B_MODELS[i][n] = b_next

    ENERGY_FD[n] = calculate_energy(E_FD[n], B_FD[n], X)
    for i in range(len(MODELS)):
        ENERGY_MODELS[i][n] = calculate_energy(E_MODELS[i][n], B_MODELS[i][n], X)

print("Rollout done.")


# ===========================
# FIGURE HELPERS
# ===========================
def save_spacetime_multicol(FD_FIELD, MODEL_FIELDS, TITLE, FILENAME_PREFIX, CMAP):
    arrays_for_limits = [FD_FIELD] + MODEL_FIELDS
    vmin, vmax = symm_limits(*arrays_for_limits)

    ncols = 1 + len(MODEL_FIELDS)
    fig, axes = matplotlib_pyplot.subplots(1, ncols, figsize=(5.2 * ncols, 4), constrained_layout=True)
    extent = [X[0], X[-1], TIME[0], TIME[-1]]

    im0 = axes[0].imshow(FD_FIELD, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=CMAP)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")

    last_im = im0
    for i, arr in enumerate(MODEL_FIELDS):
        ax = axes[i + 1]
        last_im = ax.imshow(arr, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=CMAP)
        ax.set_title(MODELS[i]["label"])
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.95)
    cbar.set_label(TITLE)

    fig.suptitle(f"{PDE}: {TITLE} (space–time)", y=1.05)
    png_path = os.path.join(SAVE_DIR, f"{FILENAME_PREFIX}.png")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    matplotlib_pyplot.close(fig)

def save_spacetime_error_multicol(ERROR_FIELDS, TITLE, FILENAME_PREFIX, CMAP):
    vmin, vmax = symm_limits(*ERROR_FIELDS)

    ncols = len(ERROR_FIELDS)
    fig, axes = matplotlib_pyplot.subplots(1, ncols, figsize=(5.2 * ncols, 4), constrained_layout=True)
    if ncols == 1:
        axes = [axes]

    extent = [X[0], X[-1], TIME[0], TIME[-1]]

    last_im = None
    for i, err in enumerate(ERROR_FIELDS):
        ax = axes[i]
        last_im = ax.imshow(err, origin="lower", aspect="auto", extent=extent, vmin=vmin, vmax=vmax, cmap=CMAP)
        ax.set_title(f"{MODELS[i]['label']} - Ground Truth")
        ax.set_xlabel("x")
        ax.set_ylabel("t")

    cbar = fig.colorbar(last_im, ax=axes, shrink=0.95)
    cbar.set_label(TITLE)

    fig.suptitle(f"{PDE}: {TITLE} error (space–time)", y=1.05)
    png_path = os.path.join(SAVE_DIR, f"{FILENAME_PREFIX}.png")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    matplotlib_pyplot.close(fig)

def save_energy_plot():
    fig = matplotlib_pyplot.figure(figsize=(7, 4), constrained_layout=True)
    matplotlib_pyplot.plot(TIME, ENERGY_FD, label="Ground Truth")
    for i in range(len(MODELS)):
        matplotlib_pyplot.plot(TIME, ENERGY_MODELS[i], label=MODELS[i]["label"])
    matplotlib_pyplot.xlabel("t")
    matplotlib_pyplot.ylabel("Energy")
    matplotlib_pyplot.title(f"{PDE}: Energy evolution")
    matplotlib_pyplot.grid(True, alpha=0.3)
    matplotlib_pyplot.legend()
    png_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_energy.png")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    matplotlib_pyplot.close(fig)

def save_relative_error_curves():
    eps = 1e-12
    fd_norm_E = np.linalg.norm(E_FD, axis=1) + eps
    fd_norm_B = np.linalg.norm(B_FD, axis=1) + eps

    fig, axes = matplotlib_pyplot.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    for i in range(len(MODELS)):
        rel_E = np.linalg.norm(E_MODELS[i] - E_FD, axis=1) / fd_norm_E
        rel_B = np.linalg.norm(B_MODELS[i] - B_FD, axis=1) / fd_norm_B
        axes[0].plot(TIME, rel_E, label=f"{MODELS[i]['label']} vs Ground Truth")
        axes[1].plot(TIME, rel_B, label=f"{MODELS[i]['label']} vs Ground Truth")

    axes[0].set_title("Relative L2 error (E field)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("||E_model - E_GT|| / ||E_GT||")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Relative L2 error (B field)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("||B_model - B_GT|| / ||B_GT||")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f"{PDE}: Relative errors vs Ground Truth")
    png_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_relative_errors.png")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    matplotlib_pyplot.close(fig)

def save_snapshots():
    snapshot_indices = [int(np.argmin(np.abs(TIME - ts))) for ts in SNAPSHOT_TIMES]
    ncols = len(snapshot_indices)

    fig, axes = matplotlib_pyplot.subplots(2, ncols, figsize=(4.0 * ncols, 6.0), constrained_layout=True)

    for j, idx in enumerate(snapshot_indices):
        axes[0, j].plot(X, E_FD[idx], label="Ground Truth")
        for i in range(len(MODELS)):
            axes[0, j].plot(X, E_MODELS[i][idx], label=MODELS[i]["label"])
        axes[0, j].set_title(f"E(x,t) at t≈{TIME[idx]:.3f}")
        axes[0, j].set_xlabel("x")
        axes[0, j].grid(True, alpha=0.3)

        axes[1, j].plot(X, B_FD[idx], label="Ground Truth")
        for i in range(len(MODELS)):
            axes[1, j].plot(X, B_MODELS[i][idx], label=MODELS[i]["label"])
        axes[1, j].set_title(f"B(x,t) at t≈{TIME[idx]:.3f}")
        axes[1, j].set_xlabel("x")
        axes[1, j].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Electric field E")
    axes[1, 0].set_ylabel("Magnetic field B")
    axes[0, -1].legend(loc="upper right")

    fig.suptitle(f"{PDE}: Snapshots")
    png_path = os.path.join(SAVE_DIR, f"{safe_name(PDE)}_snapshots.png")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    matplotlib_pyplot.close(fig)


# ===========================
# MAKE FIGURES
# ===========================
NAME = safe_name(PDE)

# Space-time fields
save_spacetime_multicol(
    FD_FIELD=E_FD,
    MODEL_FIELDS=[E_MODELS[i] for i in range(len(MODELS))],
    TITLE="Electric field E",
    FILENAME_PREFIX=f"{NAME}_spacetime_E",
    CMAP=FIELD_CMAP,
)
save_spacetime_multicol(
    FD_FIELD=B_FD,
    MODEL_FIELDS=[B_MODELS[i] for i in range(len(MODELS))],
    TITLE="Magnetic field B",
    FILENAME_PREFIX=f"{NAME}_spacetime_B",
    CMAP=FIELD_CMAP,
)

# Errors
ERR_E_LIST = [E_MODELS[i] - E_FD for i in range(len(MODELS))]
ERR_B_LIST = [B_MODELS[i] - B_FD for i in range(len(MODELS))]

save_spacetime_error_multicol(
    ERROR_FIELDS=ERR_E_LIST,
    TITLE="E field error",
    FILENAME_PREFIX=f"{NAME}_spacetime_error_E",
    CMAP=ERROR_CMAP,
)
save_spacetime_error_multicol(
    ERROR_FIELDS=ERR_B_LIST,
    TITLE="B field error",
    FILENAME_PREFIX=f"{NAME}_spacetime_error_B",
    CMAP=ERROR_CMAP,
)

save_energy_plot()
save_relative_error_curves()
save_snapshots()

# Save raw arrays
np.savez(
    os.path.join(SAVE_DIR, f"{NAME}_rollout_data.npz"),
    x=X,
    time=TIME,
    E_fd=E_FD, B_fd=B_FD,
    **{f"E_{MODELS[i]['label']}": E_MODELS[i] for i in range(len(MODELS))},
    **{f"B_{MODELS[i]['label']}": B_MODELS[i] for i in range(len(MODELS))},
    energy_fd=ENERGY_FD,
    **{f"energy_{MODELS[i]['label']}": ENERGY_MODELS[i] for i in range(len(MODELS))},
)

print("Saved paper figures to:", SAVE_DIR)
