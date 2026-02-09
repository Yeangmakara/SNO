# DeepONet.py
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

# DeepONet-specific hyperparameters
LATENT_DIM = 128                 # "p" in many DeepONet papers
BRANCH_HIDDEN_DIMS = [500, 500, 500, 500, 500]  # MLP widths for branch net
TRUNK_HIDDEN_DIMS = [500, 500, 500, 500]   # MLP widths for trunk net
ACTIVATION = "gelu"              # "relu" or "gelu" etc.

# Coordinate domain for trunk net
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


class DeepONet1d(torch.nn.Module):
    """
    Deep Operator Network for 1D function -> 1D function mapping.

    Input:
        x: (B, IN_CHANNELS, NX)
    Output:
        y: (B, OUT_CHANNELS, NX)

    Mechanism:
      - Branch net encodes the input function u into coefficients (B, OUT_CHANNELS, LATENT_DIM)
      - Trunk net encodes coordinates s into basis values (NX, LATENT_DIM)
      - Output is dot-product over LATENT_DIM at each coordinate:
            y[b, c, n] = sum_k branch[b, c, k] * trunk[n, k] + bias[c]
    """

    def __init__(
        self,
        NX,
        IN_CHANNELS,
        OUT_CHANNELS,
        LATENT_DIM,
        BRANCH_HIDDEN_DIMS,
        TRUNK_HIDDEN_DIMS,
        ACTIVATION,
        X_MIN,
        X_MAX,
    ):
        super().__init__()
        self.NX = int(NX)
        self.IN_CHANNELS = int(IN_CHANNELS)
        self.OUT_CHANNELS = int(OUT_CHANNELS)
        self.LATENT_DIM = int(LATENT_DIM)

        branch_input_dim = self.IN_CHANNELS * self.NX
        branch_output_dim = self.OUT_CHANNELS * self.LATENT_DIM
        self.branch_net = build_mlp(
            INPUT_DIM=branch_input_dim,
            HIDDEN_DIMS=BRANCH_HIDDEN_DIMS,
            OUTPUT_DIM=branch_output_dim,
            ACTIVATION=ACTIVATION,
        )

        trunk_input_dim = 1
        trunk_output_dim = self.LATENT_DIM
        self.trunk_net = build_mlp(
            INPUT_DIM=trunk_input_dim,
            HIDDEN_DIMS=TRUNK_HIDDEN_DIMS,
            OUTPUT_DIM=trunk_output_dim,
            ACTIVATION=ACTIVATION,
        )

        coords = torch.linspace(float(X_MIN), float(X_MAX), steps=self.NX).unsqueeze(-1)  # (NX, 1)
        self.register_buffer("coords", coords)

        self.output_bias = torch.nn.Parameter(torch.zeros(1, self.OUT_CHANNELS, 1))

    def forward(self, x):
        # x: (B, IN_CHANNELS, NX)
        batch_size = x.shape[0]

        # Branch: flatten function sample
        branch_in = x.reshape(batch_size, self.IN_CHANNELS * self.NX)  # (B, IN_CHANNELS*NX)
        branch_out = self.branch_net(branch_in)                        # (B, OUT_CHANNELS*LATENT_DIM)
        branch_out = branch_out.reshape(batch_size, self.OUT_CHANNELS, self.LATENT_DIM)

        # Trunk: basis at coordinates
        trunk_out = self.trunk_net(self.coords)                        # (NX, LATENT_DIM)

        # Dot product over LATENT_DIM -> (B, OUT_CHANNELS, NX)
        y = torch.einsum("bcl,nl->bcn", branch_out, trunk_out)
        y = y + self.output_bias
        return y


def get_deeponet_model(
    NX,
    IN_CHANNELS,
    OUT_CHANNELS,
    LATENT_DIM,
    BRANCH_HIDDEN_DIMS,
    TRUNK_HIDDEN_DIMS,
    ACTIVATION,
    X_MIN,
    X_MAX,
):
    model = DeepONet1d(
        NX=NX,
        IN_CHANNELS=IN_CHANNELS,
        OUT_CHANNELS=OUT_CHANNELS,
        LATENT_DIM=LATENT_DIM,
        BRANCH_HIDDEN_DIMS=BRANCH_HIDDEN_DIMS,
        TRUNK_HIDDEN_DIMS=TRUNK_HIDDEN_DIMS,
        ACTIVATION=ACTIVATION,
        X_MIN=X_MIN,
        X_MAX=X_MAX,
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

    model = get_deeponet_model(
        NX=NX,
        IN_CHANNELS=IN_CHANNELS,
        OUT_CHANNELS=OUT_CHANNELS,
        LATENT_DIM=LATENT_DIM,
        BRANCH_HIDDEN_DIMS=BRANCH_HIDDEN_DIMS,
        TRUNK_HIDDEN_DIMS=TRUNK_HIDDEN_DIMS,
        ACTIVATION=ACTIVATION,
        X_MIN=X_MIN,
        X_MAX=X_MAX,
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
            checkpoint_path = os.path.join(SAVE_DIR, f"DeepONet_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
            print(f"Elapsed time after epoch {epoch}: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print("Training complete")
