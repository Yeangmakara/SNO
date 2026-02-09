# test9_schrodinger.py
import os
import numpy as numpy
import torch
import scipy.integrate

import FNO as model_fno
import SNO as model_sno
import GNO as model_gno
import CNO as model_cno


# ===========================
# USER-EDITABLE SETTINGS
# ===========================
PDE = "Schrodinger Equation (split into Re/Im)"

# Domain / discretization
I = [0.0, 1.0]
NX = 100
DT = 0.001

# Coefficient (matches your sch-equation.py: "*0.1")
SCH_COEFF = 0.1

# Evaluation steps (time steps)
EVAL_STEPS = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
              200, 300, 400, 500, 600, 700, 800, 900, 1000]
MAX_STEPS = max(EVAL_STEPS)

# Monte Carlo trials
N_TRIALS = 10
SEED = 0
MAX_CHEBYSHEV_DEGREE = 20

# Energy metric mode
USE_RELATIVE_ENERGY_ERROR = True

# Checkpoints
CHECKPOINT_FNO = "checkpoints/FNO_epoch_500.pth"
CHECKPOINT_SNO = "checkpoints/SNO_epoch_500.pth"
CHECKPOINT_GNO = "checkpoints/GNO_epoch_500.pth"
CHECKPOINT_CNO = "checkpoints/CNO_epoch_500.pth"

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

numpy.random.seed(SEED)
torch.manual_seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

X = numpy.linspace(I[0], I[1], NX)
DX = float(X[1] - X[0])


# ===========================
# SCHRODINGER DISCRETIZATION
# ===========================
def laplacian_matrix(N, dx):
    D2 = -2.0 * numpy.eye(N, dtype=numpy.float64)
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
    cheb_coeffs = numpy.random.uniform(-1.0, 1.0, size=MAX_CHEBYSHEV_DEGREE + 1)
    cheb_polys = [numpy.polynomial.chebyshev.Chebyshev.basis(i)(x_mapped) for i in range(len(cheb_coeffs))]
    func = sum(CC * P * (X_ARRAY - I[0]) * (I[1] - X_ARRAY) for CC, P in zip(cheb_coeffs, cheb_polys))
    func /= numpy.max(numpy.abs(func)) + 1e-12
    return func

def solve_ivp_fd_one_step(U_RE_NOW, U_IM_NOW, DT_STEP):
    """
    Matches your sch-equation.py:
        du_re = -D2 @ u_im * 0.1
        du_im =  D2 @ u_re * 0.1
    """
    nx_local = len(U_RE_NOW)

    def sch_eq(_t, y):
        u_re = y[:nx_local]
        u_im = y[nx_local:]
        du_re = -(D2 @ u_im) * SCH_COEFF
        du_im =  (D2 @ u_re) * SCH_COEFF
        return numpy.concatenate([du_re, du_im])

    y0 = numpy.concatenate([U_RE_NOW, U_IM_NOW])
    sol = scipy.integrate.solve_ivp(
        sch_eq,
        [0.0, DT_STEP],
        y0,
        method="RK45",
        rtol=1e-8,
        atol=1e-8,
    )
    y_end = sol.y[:, -1]
    u_re_new = y_end[:nx_local]
    u_im_new = y_end[nx_local:]
    return u_re_new, u_im_new

def model_step(MODEL, U_RE_ARR, U_IM_ARR):
    inp = torch.tensor(numpy.stack([U_RE_ARR, U_IM_ARR], axis=0), dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        pred = MODEL(inp).detach().cpu().squeeze().numpy()  # (2, NX)
    return pred[0], pred[1]

def l2_state_error(U_RE_MODEL, U_IM_MODEL, U_RE_GT, U_IM_GT):
    return float(numpy.sqrt(numpy.linalg.norm(U_RE_MODEL - U_RE_GT) ** 2 + numpy.linalg.norm(U_IM_MODEL - U_IM_GT) ** 2))

def calculate_energy(U_RE_ARR, U_IM_ARR, X_ARR):
    dx_local = float(X_ARR[1] - X_ARR[0])
    # gradient with spacing (your code: np.gradient(f, dx))
    d_re = numpy.gradient(U_RE_ARR, dx_local)
    d_im = numpy.gradient(U_IM_ARR, dx_local)
    energy_density = 0.5 * (numpy.abs(d_re) ** 2 + numpy.abs(d_im) ** 2)
    return float(numpy.sum(energy_density) * dx_local * SCH_COEFF)

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
    U_RE0 = random_func_T(X) / 10.0
    U_IM0 = random_func_T(X) / 10.0

    # Ground truth state
    U_RE_FD = U_RE0.copy()
    U_IM_FD = U_IM0.copy()

    # Model states
    U_RE_STATE = [U_RE0.copy() for _ in MODELS]
    U_IM_STATE = [U_IM0.copy() for _ in MODELS]

    for STEP in range(1, MAX_STEPS + 1):
        U_RE_FD, U_IM_FD = solve_ivp_fd_one_step(U_RE_FD, U_IM_FD, DT_STEP=DT)

        for i, entry in enumerate(MODELS):
            re_next, im_next = model_step(entry["model"], U_RE_STATE[i], U_IM_STATE[i])
            U_RE_STATE[i] = re_next
            U_IM_STATE[i] = im_next

        if STEP in EVAL_STEPS:
            E_GT = calculate_energy(U_RE_FD, U_IM_FD, X)

            for i, entry in enumerate(MODELS):
                label = entry["label"]

                l2 = l2_state_error(U_RE_STATE[i], U_IM_STATE[i], U_RE_FD, U_IM_FD)
                L2_ERRORS[label][STEP].append(l2)

                E_M = calculate_energy(U_RE_STATE[i], U_IM_STATE[i], X)
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
            mean_dict[label][step] = float(arr.mean())
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


L2_HEADER, L2_ROWS = build_rows(L2_MEAN, L2_STD)
EN_HEADER, EN_ROWS = build_rows(EN_MEAN, EN_STD)

print_table("Table 1: Average L2 state error  (mean ± std over trials)", L2_HEADER, L2_ROWS)

metric_name = "relative energy error" if USE_RELATIVE_ENERGY_ERROR else "absolute energy error"
print_table("Table 2: Average " + metric_name + "  (mean ± std over trials)", EN_HEADER, EN_ROWS)

CSV_L2_PATH = os.path.join(SAVE_DIR, CSV_L2)
CSV_EN_PATH = os.path.join(SAVE_DIR, CSV_ENERGY)
save_csv(CSV_L2_PATH, L2_HEADER, L2_ROWS)
save_csv(CSV_EN_PATH, EN_HEADER, EN_ROWS)
print("\nSaved CSV:", CSV_L2_PATH)
print("Saved CSV:", CSV_EN_PATH)

TEX_L2_PATH = os.path.join(SAVE_DIR, TEX_L2)
TEX_EN_PATH = os.path.join(SAVE_DIR, TEX_ENERGY)

save_latex(
    TEX_L2_PATH,
    "Average $L_2$ error of the state $[re;im]$ at selected rollout steps (mean $\\pm$ std over {} random initial conditions).".format(N_TRIALS),
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
