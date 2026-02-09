# test9_klein_gordon.py
import os
import numpy as np
import torch
import scipy.integrate

import FNO as model_fno
import SNO as model_sno
import GNO as model_gno
import CNO as model_cno


# ===========================
# USER-EDITABLE SETTINGS
# ===========================
PDE = "Nonlinear Klein-Gordon Equation"
#  u_tt = c^2 u_xx + u + u^3 (Dirichlet BC)"

# Domain / discretization
I = [0.0, 1.0]
C_KG = 0.1 
NX = 100
DT = 0.001

# Evaluation steps (time steps)
EVAL_STEPS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
              200, 300, 400, 500, 600, 700, 800, 900, 1000,
              1500, 2000, 2500, 3000]
MAX_STEPS = max(EVAL_STEPS)

# Monte Carlo trials
N_TRIALS = 10
SEED = 0
MAX_CHEBYSHEV_DEGREE = 20

# Energy metric mode
USE_RELATIVE_ENERGY_ERROR = True   # True: |E_model - E_GT|/(|E_GT|+eps), False: |E_model - E_GT|

# Checkpoints
CHECKPOINT_FNO = "checkpoints/FNO_epoch_100.pth"
CHECKPOINT_SNO = "checkpoints/SNO_epoch_100.pth"
CHECKPOINT_GNO = "checkpoints/GNO_epoch_100.pth"
CHECKPOINT_CNO = "checkpoints/CNO_epoch_100.pth"

# Labels (shown in table)
LABEL_SNO = "SNO (ours)"
LABEL_FNO = "FNO"
LABEL_GNO = "GNO"
LABEL_CNO = "CNO"

# Output
SAVE_DIR = "eval_tables_both"
CSV_L2 = "table_l2.csv"
CSV_ENERGY = "table_energy.csv"
TEX_L2 = "table_l2.tex"
TEX_ENERGY = "table_energy.tex"

# ---- GNO hyperparameters (MUST match training) ----
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

# ---- CNO hyperparameters (MUST match training) ----
CNO_IN_CHANNELS = 2
CNO_OUT_CHANNELS = 2
CNO_WIDTH = 128
CNO_DEPTH = 5
CNO_KERNEL_SIZE = 5
CNO_USE_CIRCULAR_PADDING = False


# ===========================
# DEVICE / RNG
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

np.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

X = np.linspace(I[0], I[1], NX)
DX = X[1] - X[0]


# ===========================
# KLEIN-GORDON DISCRETIZATION
# ===========================
def laplacian_matrix(N, dx):
    # Dirichlet BCs, same structure as your klein-gordon.py
    D2 = -2.0 * np.eye(N, dtype=np.float64)
    for i in range(1, N - 1):
        D2[i, i - 1] = 1.0
        D2[i, i + 1] = 1.0
    D2[0, :] = 0.0
    D2[-1, :] = 0.0
    return D2 / (dx ** 2)

D2 = laplacian_matrix(int(NX), float(DX))


# ===========================
# HELPERS
# ===========================
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

def solve_ivp_fd_one_step(F_NOW, G_NOW, DT_STEP):
    """
    Nonlinear Klein-Gordon one step:
        u_tt = c^2 u_xx + u + u^3
    with Dirichlet BCs enforced in dp.
    """
    nx_local = len(F_NOW)

    def kg_eq(_t, y):
        q = y[:nx_local]
        p = y[nx_local:]

        dq = p
        dp = (C_KG ** 2) * (D2 @ q) + q + (q ** 3)

        # Dirichlet BCs (same as your klein-gordon.py)
        dp[0] = 0.0
        dp[-1] = 0.0
        return np.concatenate([dq, dp])

    y0 = np.concatenate([F_NOW, G_NOW])
    sol = scipy.integrate.solve_ivp(
        kg_eq,
        [0.0, DT_STEP],
        y0,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    y_end = sol.y[:, -1]
    f_new = y_end[:nx_local]
    g_new = y_end[nx_local:]
    return f_new, g_new

def model_step(MODEL, F_ARR, G_ARR):
    inp = torch.tensor(np.stack([F_ARR, G_ARR], axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = MODEL(inp).detach().cpu().squeeze().numpy()  # (2, NX)
    return pred[0], pred[1]

def l2_state_error(F_MODEL, G_MODEL, F_GT, G_GT):
    # absolute L2 error of stacked state [f; g]
    return float(np.sqrt(np.linalg.norm(F_MODEL - F_GT) ** 2 + np.linalg.norm(G_MODEL - G_GT) ** 2))

def calculate_energy(F_ARR, G_ARR, X_ARR):
    """
    Energy functional for nonlinear Klein-Gordon (matching your klein-gordon.py):
        E = ∫ [ 0.5 g^2 + 0.5 c^2 (u_x)^2 - 0.5 u^2 - 0.25 u^4 ] dx
    """
    # Use grid spacing consistent with X_ARR
    dx_local = float(X_ARR[1] - X_ARR[0])
    du_dx = np.gradient(F_ARR, dx_local)
    energy_density = 0.5 * (G_ARR ** 2) + 0.5 * (C_KG ** 2) * (du_dx ** 2) - 0.5 * (F_ARR ** 2) - 0.25 * (F_ARR ** 4)
    return float(np.sum(energy_density) * dx_local)

def energy_error(E_MODEL, E_GT):
    eps = 1e-12
    if USE_RELATIVE_ENERGY_ERROR:
        return abs(E_MODEL - E_GT) / (abs(E_GT) + eps)
    return abs(E_MODEL - E_GT)


# ===========================
# BUILD + LOAD MODELS
# ===========================
MODEL_FNO = model_fno.get_fno_model()
SD_FNO = strip_module_prefix(torch.load(CHECKPOINT_FNO, map_location=DEVICE))
MODEL_FNO.load_state_dict(SD_FNO, strict=True)
MODEL_FNO.to(DEVICE).eval()

MODEL_SNO = model_sno.get_model()
SD_SNO = strip_module_prefix(torch.load(CHECKPOINT_SNO, map_location=DEVICE))
MODEL_SNO.load_state_dict(SD_SNO, strict=True)
MODEL_SNO.to(DEVICE).eval()

MODEL_GNO = model_gno.get_gno_model(
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

MODEL_CNO = model_cno.get_cno_model(
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
# MONTE CARLO EVALUATION
# ===========================
L2_ERRORS = {m["label"]: {s: [] for s in EVAL_STEPS} for m in MODELS}
ENERGY_ERRORS = {m["label"]: {s: [] for s in EVAL_STEPS} for m in MODELS}

print(f"PDE: {PDE}")
print(f"Running {N_TRIALS} trials, evaluating at steps {EVAL_STEPS} (max rollout {MAX_STEPS}) ...")

for TRIAL in range(N_TRIALS):
    # Random initial condition
    F0 = random_func_T(X) / 3.0
    G0 = random_func_T(X) / 3.0

    # (Optional) enforce Dirichlet BC at t=0 for the GT state
    F0[0] = 0.0
    F0[-1] = 0.0

    # Ground truth state
    F_FD = F0.copy()
    G_FD = G0.copy()

    # Model states
    F_STATE = [F0.copy() for _ in MODELS]
    G_STATE = [G0.copy() for _ in MODELS]

    for STEP in range(1, MAX_STEPS + 1):
        # Ground truth one step (nonlinear KG)
        F_FD, G_FD = solve_ivp_fd_one_step(F_FD, G_FD, DT_STEP=DT)

        # Models one step
        for i, entry in enumerate(MODELS):
            f_next, g_next = model_step(entry["model"], F_STATE[i], G_STATE[i])
            F_STATE[i] = f_next
            G_STATE[i] = g_next

        if STEP in EVAL_STEPS:
            # GT energy once
            E_GT = calculate_energy(F_FD, G_FD, X)

            for i, entry in enumerate(MODELS):
                label = entry["label"]

                # L2 state error
                l2 = l2_state_error(F_STATE[i], G_STATE[i], F_FD, G_FD)
                L2_ERRORS[label][STEP].append(l2)

                # Energy error
                E_M = calculate_energy(F_STATE[i], G_STATE[i], X)
                eerr = energy_error(E_M, E_GT)
                ENERGY_ERRORS[label][STEP].append(eerr)

    print(f"  Trial {TRIAL + 1}/{N_TRIALS} done.")


# ===========================
# SUMMARIZE: mean ± std
# ===========================
def summarize(error_dict):
    mean_dict = {label: {} for label in error_dict}
    std_dict = {label: {} for label in error_dict}
    for label in error_dict:
        for step in EVAL_STEPS:
            arr = numpy.array(error_dict[label][step], dtype=numpy.float64)
            mean_dict[label][step] = float(arr.mean()) if len(arr) > 0 else float("nan")
            std_dict[label][step] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return mean_dict, std_dict

L2_MEAN, L2_STD = summarize(L2_ERRORS)
EN_MEAN, EN_STD = summarize(ENERGY_ERRORS)


# ===========================
# TABLE OUTPUT (rows=time, cols=models)
# ===========================
def format_pm(MEAN, STD):
    return f"{MEAN:.4e} ± {STD:.4e}"

def build_rows(mean_dict, std_dict):
    header = ["Time step"] + [m["label"] for m in MODELS]
    rows = []
    for step in EVAL_STEPS:
        row = [str(step)]
        for m in MODELS:
            label = m["label"]
            row.append(format_pm(mean_dict[label][step], std_dict[label][step]))
        rows.append(row)
    return header, rows

def print_table(title, header, rows):
    col_widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    line = " | ".join(h.ljust(w) for h, w in zip(header, col_widths))
    sep = "-+-".join("-" * w for w in col_widths)
    print("\n" + title + "\n")
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(str(x).ljust(w) for x, w in zip(r, col_widths)))

def save_csv(path, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")

def save_latex(path, caption, label, header, rows):
    latex_lines = []
    latex_lines.append("\\begin{table}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\small")
    latex_lines.append("\\setlength{\\tabcolsep}{6pt}")
    latex_lines.append("\\begin{tabular}{c" + "c" * (len(header) - 1) + "}")
    latex_lines.append("\\hline")
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\hline")
    for r in rows:
        latex_lines.append(" & ".join(r) + " \\\\")
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\caption{" + caption + "}")
    latex_lines.append("\\label{" + label + "}")
    latex_lines.append("\\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(latex_lines) + "\n")


# Build both tables
L2_HEADER, L2_ROWS = build_rows(L2_MEAN, L2_STD)
EN_HEADER, EN_ROWS = build_rows(EN_MEAN, EN_STD)

# Print both tables
print_table("Table 1: Average L2 state error  (mean ± std over trials)", L2_HEADER, L2_ROWS)

metric_name = "relative energy error" if USE_RELATIVE_ENERGY_ERROR else "absolute energy error"
print_table("Table 2: Average " + metric_name + "  (mean ± std over trials)", EN_HEADER, EN_ROWS)

# Save both CSV
CSV_L2_PATH = os.path.join(SAVE_DIR, CSV_L2)
CSV_EN_PATH = os.path.join(SAVE_DIR, CSV_ENERGY)
save_csv(CSV_L2_PATH, L2_HEADER, L2_ROWS)
save_csv(CSV_EN_PATH, EN_HEADER, EN_ROWS)
print("\nSaved CSV:", CSV_L2_PATH)
print("Saved CSV:", CSV_EN_PATH)

# Save both LaTeX
TEX_L2_PATH = os.path.join(SAVE_DIR, TEX_L2)
TEX_EN_PATH = os.path.join(SAVE_DIR, TEX_ENERGY)

save_latex(
    TEX_L2_PATH,
    "Average $\\ell_2$ error of the state $[f;g]$ at selected rollout steps (mean $\\pm$ std over {} random initial conditions).".format(N_TRIALS),
    "tab:l2_errors_time_rows",
    L2_HEADER,
    L2_ROWS,
)

save_latex(
    TEX_EN_PATH,
    "Average " + metric_name + " at selected rollout steps (mean $\\pm$ std over {} random initial conditions).".format(N_TRIALS),
    "tab:energy_errors_time_rows",
    EN_HEADER,
    EN_ROWS,
)

print("Saved LaTeX table:", TEX_L2_PATH)
print("Saved LaTeX table:", TEX_EN_PATH)
