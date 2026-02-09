# CNO.py
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

# CNO-specific hyperparameters
WIDTH = 128
DEPTH = 5                 # number of down/up levels
KERNEL_SIZE = 5
USE_CIRCULAR_PADDING = False  # set True if your PDE domain is periodic

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


class ConvBlock1d(torch.nn.Module):
    def __init__(self, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, USE_CIRCULAR_PADDING):
        super().__init__()
        padding = int(KERNEL_SIZE // 2)
        padding_mode = "circular" if bool(USE_CIRCULAR_PADDING) else "zeros"

        self.conv1 = torch.nn.Conv1d(
            in_channels=int(IN_CHANNELS),
            out_channels=int(OUT_CHANNELS),
            kernel_size=int(KERNEL_SIZE),
            padding=padding,
            padding_mode=padding_mode,
        )
        self.conv2 = torch.nn.Conv1d(
            in_channels=int(OUT_CHANNELS),
            out_channels=int(OUT_CHANNELS),
            kernel_size=int(KERNEL_SIZE),
            padding=padding,
            padding_mode=padding_mode,
        )
        self.norm1 = torch.nn.GroupNorm(num_groups=8, num_channels=int(OUT_CHANNELS))
        self.norm2 = torch.nn.GroupNorm(num_groups=8, num_channels=int(OUT_CHANNELS))

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = torch.nn.functional.gelu(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = torch.nn.functional.gelu(x)
        return x


class CNO1d(torch.nn.Module):
    """
    Convolutional Neural Operator (practical 1D multiresolution conv operator / U-Net style).

    Input:  (B, IN_CHANNELS, NX)
    Output: (B, OUT_CHANNELS, NX)
    """

    def __init__(self, IN_CHANNELS, OUT_CHANNELS, WIDTH, DEPTH, KERNEL_SIZE, USE_CIRCULAR_PADDING):
        super().__init__()
        self.DEPTH = int(DEPTH)

        self.lift = torch.nn.Conv1d(int(IN_CHANNELS), int(WIDTH), kernel_size=1)

        self.down_blocks = torch.nn.ModuleList()
        self.downsamplers = torch.nn.ModuleList()

        channels = int(WIDTH)
        for _ in range(self.DEPTH):
            self.down_blocks.append(
                ConvBlock1d(
                    IN_CHANNELS=channels,
                    OUT_CHANNELS=channels,
                    KERNEL_SIZE=KERNEL_SIZE,
                    USE_CIRCULAR_PADDING=USE_CIRCULAR_PADDING,
                )
            )
            self.downsamplers.append(torch.nn.AvgPool1d(kernel_size=2, stride=2))

        self.bottleneck = ConvBlock1d(
            IN_CHANNELS=channels,
            OUT_CHANNELS=channels,
            KERNEL_SIZE=KERNEL_SIZE,
            USE_CIRCULAR_PADDING=USE_CIRCULAR_PADDING,
        )

        self.up_blocks = torch.nn.ModuleList()
        for _ in range(self.DEPTH):
            self.up_blocks.append(
                ConvBlock1d(
                    IN_CHANNELS=2 * channels,  # concat skip
                    OUT_CHANNELS=channels,
                    KERNEL_SIZE=KERNEL_SIZE,
                    USE_CIRCULAR_PADDING=USE_CIRCULAR_PADDING,
                )
            )

        self.project = torch.nn.Conv1d(int(WIDTH), int(OUT_CHANNELS), kernel_size=1)

    def forward(self, x):
        # x: (B, IN_CHANNELS, NX)
        x = self.lift(x)

        skips = []
        h = x

        # Encoder
        for level in range(self.DEPTH):
            h = self.down_blocks[level](h)
            skips.append(h)
            h = self.downsamplers[level](h)

        # Bottleneck
        h = self.bottleneck(h)

        # Decoder
        for level in range(self.DEPTH - 1, -1, -1):
            # Upsample to match the skip length exactly
            target_len = int(skips[level].shape[-1])
            h = torch.nn.functional.interpolate(h, size=target_len, mode="linear", align_corners=False)

            h = torch.cat([h, skips[level]], dim=1)  # concat channels
            h = self.up_blocks[self.DEPTH - 1 - level](h)

        y = self.project(h)
        return y


def get_cno_model(IN_CHANNELS, OUT_CHANNELS, WIDTH, DEPTH, KERNEL_SIZE, USE_CIRCULAR_PADDING):
    model = CNO1d(
        IN_CHANNELS=IN_CHANNELS,
        OUT_CHANNELS=OUT_CHANNELS,
        WIDTH=WIDTH,
        DEPTH=DEPTH,
        KERNEL_SIZE=KERNEL_SIZE,
        USE_CIRCULAR_PADDING=USE_CIRCULAR_PADDING,
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

    model = get_cno_model(
        IN_CHANNELS=IN_CHANNELS,
        OUT_CHANNELS=OUT_CHANNELS,
        WIDTH=WIDTH,
        DEPTH=DEPTH,
        KERNEL_SIZE=KERNEL_SIZE,
        USE_CIRCULAR_PADDING=USE_CIRCULAR_PADDING,
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
            checkpoint_path = os.path.join(SAVE_DIR, f"CNO_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
            print(f"Elapsed time after epoch {epoch}: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print("Training complete")
