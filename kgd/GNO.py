# GNO.py
import os
import time
import numpy
import torch
import torch.utils.data
import torch.nn.functional
import sklearn.model_selection

# -------------------------
# User-changeable constants
# -------------------------
DATA_PATH = "data/data.npz"
TEST_SIZE = 0.2
RANDOM_STATE = 42

BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-3

SAVE_INTERVAL = 5
SAVE_DIR = "checkpoints"

# Model hyperparameters
IN_CHANNELS = 2
OUT_CHANNELS = 2

# GNO-specific hyperparameters
HIDDEN_CHANNELS = 256
NUM_LAYERS = 7

NEIGHBOR_RADIUS = 8        # number of neighbors on each side
AGGREGATION = "mean"       # "mean" or "sum"
PERIODIC = False           # set True if your domain is periodic

# Kernel MLP hyperparameters (maps delta_x -> per-channel weights)
KERNEL_HIDDEN_DIMS = [256, 256, 256]
KERNEL_ACTIVATION = "gelu"

# Coordinate domain for building the graph
X_MIN = 0.0
X_MAX = 1.0

# Automatically select GPU if available, otherwise use CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


def load_wave_data(FILE_PATH, TEST_SIZE, RANDOM_STATE):
    data = numpy.load(FILE_PATH)
    inputs = torch.from_numpy(data["inputs"]).float()
    targets = torch.from_numpy(data["targets"]).float()

    NX = inputs.shape[1] // 2
    inputs = inputs.reshape(-1, 2, NX)
    targets = targets.reshape(-1, 2, NX)

    inputs_train, inputs_test, targets_train, targets_test = sklearn.model_selection.train_test_split(
        inputs, targets, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return inputs_train, targets_train, inputs_test, targets_test


def build_mlp(INPUT_DIM, HIDDEN_DIMS, OUTPUT_DIM, ACTIVATION):
    layers = []
    last_dim = INPUT_DIM

    for hidden_dim in HIDDEN_DIMS:
        layers.append(torch.nn.Linear(last_dim, hidden_dim))
        if ACTIVATION.lower() == "relu":
            layers.append(torch.nn.ReLU())
        elif ACTIVATION.lower() == "tanh":
            layers.append(torch.nn.Tanh())
        else:
            layers.append(torch.nn.GELU())
        last_dim = hidden_dim

    layers.append(torch.nn.Linear(last_dim, OUTPUT_DIM))
    return torch.nn.Sequential(*layers)


def make_neighbor_index_and_deltas(NX, NEIGHBOR_RADIUS, X_MIN, X_MAX, PERIODIC):
    """
    Builds:
      neighbor_index: (NX, K) int64
      delta_x:        (NX, K, 1) float32

    K = 2 * NEIGHBOR_RADIUS (excluding self)
    """
    NX = int(NX)
    R = int(NEIGHBOR_RADIUS)
    assert R >= 1

    coords = torch.linspace(float(X_MIN), float(X_MAX), steps=NX)  # (NX,)
    K = 2 * R
    neighbor_index = torch.empty((NX, K), dtype=torch.long)
    delta_x = torch.empty((NX, K, 1), dtype=torch.float32)

    offsets = [d for d in range(-R, R + 1) if d != 0]  # exclude 0
    assert len(offsets) == K

    for i in range(NX):
        for k, d in enumerate(offsets):
            j = i + d
            if PERIODIC:
                j = j % NX
            else:
                j = int(max(0, min(NX - 1, j)))

            neighbor_index[i, k] = j
            delta = coords[j] - coords[i]
            delta_x[i, k, 0] = float(delta.item())

    return coords.unsqueeze(-1), neighbor_index, delta_x


class GNOLayer1d(torch.nn.Module):
    """
    One message-passing layer with a learned kernel:
        m_i = sum_{j in N(i)} K(delta_x_ij) ⊙ f_j
    where K outputs per-hidden-channel weights and ⊙ is elementwise product.
    """

    def __init__(self, HIDDEN_CHANNELS, KERNEL_HIDDEN_DIMS, KERNEL_ACTIVATION, AGGREGATION):
        super().__init__()
        self.HIDDEN_CHANNELS = int(HIDDEN_CHANNELS)
        self.AGGREGATION = str(AGGREGATION).lower()

        # Kernel: delta_x (1-dim) -> weights (HIDDEN_CHANNELS)
        self.kernel_mlp = build_mlp(
            INPUT_DIM=1,
            HIDDEN_DIMS=KERNEL_HIDDEN_DIMS,
            OUTPUT_DIM=self.HIDDEN_CHANNELS,
            ACTIVATION=KERNEL_ACTIVATION,
        )

        self.self_linear = torch.nn.Linear(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS)
        self.msg_linear = torch.nn.Linear(self.HIDDEN_CHANNELS, self.HIDDEN_CHANNELS)
        self.norm = torch.nn.LayerNorm(self.HIDDEN_CHANNELS)

    def forward(self, f, neighbor_index, delta_x):
        """
        f: (B, NX, HIDDEN_CHANNELS)
        neighbor_index: (NX, K)
        delta_x: (NX, K, 1)
        """
        # Gather neighbor features: (B, NX, K, H)
        f_neighbors = f[:, neighbor_index, :]

        # Kernel weights: (NX, K, H)
        w = self.kernel_mlp(delta_x)

        # Broadcast to batch: (B, NX, K, H)
        w = w.unsqueeze(0)

        # Message: (B, NX, K, H)
        msg = w * f_neighbors

        if self.AGGREGATION == "sum":
            msg_agg = msg.sum(dim=2)   # (B, NX, H)
        else:
            msg_agg = msg.mean(dim=2)  # (B, NX, H)

        # Update with residual + norm
        out = self.self_linear(f) + self.msg_linear(msg_agg)
        out = torch.nn.functional.gelu(out)
        out = self.norm(out + f)
        return out


class GNO1d(torch.nn.Module):
    """
    Graph Neural Operator on a 1D grid graph.
    Input:  (B, IN_CHANNELS, NX)
    Output: (B, OUT_CHANNELS, NX)
    """

    def __init__(
        self,
        NX,
        IN_CHANNELS,
        OUT_CHANNELS,
        HIDDEN_CHANNELS,
        NUM_LAYERS,
        NEIGHBOR_RADIUS,
        KERNEL_HIDDEN_DIMS,
        KERNEL_ACTIVATION,
        AGGREGATION,
        X_MIN,
        X_MAX,
        PERIODIC,
    ):
        super().__init__()
        self.NX = int(NX)

        coords, neighbor_index, delta_x = make_neighbor_index_and_deltas(
            NX=self.NX,
            NEIGHBOR_RADIUS=NEIGHBOR_RADIUS,
            X_MIN=X_MIN,
            X_MAX=X_MAX,
            PERIODIC=PERIODIC,
        )

        self.register_buffer("coords", coords)                 # (NX, 1)
        self.register_buffer("neighbor_index", neighbor_index) # (NX, K)
        self.register_buffer("delta_x", delta_x)               # (NX, K, 1)

        self.input_linear = torch.nn.Linear(int(IN_CHANNELS), int(HIDDEN_CHANNELS))
        self.layers = torch.nn.ModuleList(
            [
                GNOLayer1d(
                    HIDDEN_CHANNELS=HIDDEN_CHANNELS,
                    KERNEL_HIDDEN_DIMS=KERNEL_HIDDEN_DIMS,
                    KERNEL_ACTIVATION=KERNEL_ACTIVATION,
                    AGGREGATION=AGGREGATION,
                )
                for _ in range(int(NUM_LAYERS))
            ]
        )
        self.output_linear = torch.nn.Linear(int(HIDDEN_CHANNELS), int(OUT_CHANNELS))

    def forward(self, x):
        # x: (B, IN_CHANNELS, NX) -> (B, NX, IN_CHANNELS)
        x = x.permute(0, 2, 1)

        f = self.input_linear(x)
        f = torch.nn.functional.gelu(f)

        for layer in self.layers:
            f = layer(f, self.neighbor_index, self.delta_x)

        y = self.output_linear(f)      # (B, NX, OUT_CHANNELS)
        y = y.permute(0, 2, 1)         # (B, OUT_CHANNELS, NX)
        return y


def get_gno_model(
    NX,
    IN_CHANNELS,
    OUT_CHANNELS,
    HIDDEN_CHANNELS,
    NUM_LAYERS,
    NEIGHBOR_RADIUS,
    KERNEL_HIDDEN_DIMS,
    KERNEL_ACTIVATION,
    AGGREGATION,
    X_MIN,
    X_MAX,
    PERIODIC,
):
    model = GNO1d(
        NX=NX,
        IN_CHANNELS=IN_CHANNELS,
        OUT_CHANNELS=OUT_CHANNELS,
        HIDDEN_CHANNELS=HIDDEN_CHANNELS,
        NUM_LAYERS=NUM_LAYERS,
        NEIGHBOR_RADIUS=NEIGHBOR_RADIUS,
        KERNEL_HIDDEN_DIMS=KERNEL_HIDDEN_DIMS,
        KERNEL_ACTIVATION=KERNEL_ACTIVATION,
        AGGREGATION=AGGREGATION,
        X_MIN=X_MIN,
        X_MAX=X_MAX,
        PERIODIC=PERIODIC,
    )
    return model.to(DEVICE)



if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)

    inputs_train, targets_train, inputs_test, targets_test = load_wave_data(
        FILE_PATH=DATA_PATH, TEST_SIZE=TEST_SIZE, RANDOM_STATE=RANDOM_STATE
    )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs_train, targets_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(inputs_test, targets_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    NX = int(inputs_train.shape[-1])

    model = get_gno_model(
        NX=NX,
        IN_CHANNELS=IN_CHANNELS,
        OUT_CHANNELS=OUT_CHANNELS,
        HIDDEN_CHANNELS=HIDDEN_CHANNELS,
        NUM_LAYERS=NUM_LAYERS,
        NEIGHBOR_RADIUS=NEIGHBOR_RADIUS,
        KERNEL_HIDDEN_DIMS=KERNEL_HIDDEN_DIMS,
        KERNEL_ACTIVATION=KERNEL_ACTIVATION,
        AGGREGATION=AGGREGATION,
        X_MIN=X_MIN,
        X_MAX=X_MAX,
        PERIODIC=PERIODIC,
    )


    total_params = sum(parameter.numel() for parameter in model.parameters())
    print("Total Parameters: ", total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / max(1, len(train_loader))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)

                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                test_loss += float(loss.item())

        avg_test_loss = test_loss / max(1, len(test_loader))

        print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {avg_loss:.10f}, Test Loss: {avg_test_loss:.10f}")

        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            elapsed_time = time.time() - start_time
            checkpoint_path = os.path.join(SAVE_DIR, f"GNO_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
            print(f"Elapsed time after epoch {epoch}: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print("Training complete")
