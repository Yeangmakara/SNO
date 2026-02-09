import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def load_wave_data(filename, test_size=0.2, random_state=42):
    data = np.load(filename)
    inputs = torch.from_numpy(data["inputs"]).float()
    targets = torch.from_numpy(data["targets"]).float()
    NX = inputs.shape[1] // 2
    inputs = inputs.reshape(-1, 2, NX)
    targets = targets.reshape(-1, 2, NX)

    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=test_size, random_state=random_state
    )
    return inputs_train, targets_train, inputs_test, targets_test


def create_mlp(INPUT_CHANNELS, OUTPUT_CHANNELS, HIDDEN_CHANNELS, ACTIVATION=nn.ReLU):
    layers = []
    prev_channels = INPUT_CHANNELS
    for channels in HIDDEN_CHANNELS:
        layers.append(nn.Conv1d(prev_channels, channels, kernel_size=1))
        layers.append(ACTIVATION())
        prev_channels = channels
    layers.append(nn.Conv1d(prev_channels, OUTPUT_CHANNELS, kernel_size=1))
    return nn.Sequential(*layers)


# ============================================================
# Self-Adjoint Linear FNO block (discrete analogue of theorem)
# ============================================================
class SelfAdjointLinearFNO1d(nn.Module):
    """
    Input:  (B, 1, N)
    Output: (B, 1, N)

    v0 = P a
    v_{k+1} = W_k v_k + F^{-1}( R_k * F v_k )
    G(a) = P^* v_L

    Enforced by construction (real setting):
      W_k = W_{L-1-k}^T
      R_k = R_{L-1-k}^* (complex conjugate-transpose on channel indices)
    """
    def __init__(self, N_MODES, WIDTH=64, N_LAYERS=4, IN_CHANNELS=1):
        super().__init__()
        self.N_MODES = int(N_MODES)
        self.WIDTH = int(WIDTH)
        self.N_LAYERS = int(N_LAYERS)
        self.IN_CHANNELS = int(IN_CHANNELS)

        # P: (m, d), keep real for simplicity (then adjoint is transpose)
        self.P = nn.Parameter(
            torch.randn(self.WIDTH, self.IN_CHANNELS) * (1.0 / math.sqrt(self.IN_CHANNELS))
        )

        half = self.N_LAYERS // 2
        self.W_free = nn.ParameterList()
        self.R_free = nn.ParameterList()

        for _ in range(half):
            self.W_free.append(
                nn.Parameter(torch.randn(self.WIDTH, self.WIDTH) * (1.0 / math.sqrt(self.WIDTH)))
            )
            # complex Fourier multipliers for low modes
            self.R_free.append(
                nn.Parameter(torch.randn(self.WIDTH, self.WIDTH, self.N_MODES, dtype=torch.cfloat) * 0.02)
            )

        # middle layer if odd: make it self-adjoint by symmetrization
        if self.N_LAYERS % 2 == 1:
            self.W_mid_raw = nn.Parameter(torch.randn(self.WIDTH, self.WIDTH) * (1.0 / math.sqrt(self.WIDTH)))
            self.R_mid_raw = nn.Parameter(torch.randn(self.WIDTH, self.WIDTH, self.N_MODES, dtype=torch.cfloat) * 0.02)
        else:
            self.W_mid_raw = None
            self.R_mid_raw = None

    def _layer_params(self, k: int):
        L = self.N_LAYERS
        half = L // 2

        # first half: free
        if k < half:
            return self.W_free[k], self.R_free[k]

        # middle
        if (L % 2 == 1) and (k == half):
            W = 0.5 * (self.W_mid_raw + self.W_mid_raw.transpose(0, 1))
            R = 0.5 * (self.R_mid_raw + self.R_mid_raw.conj().transpose(0, 1))
            return W, R

        # second half: adjoint-paired
        j = L - 1 - k
        W = self.W_free[j].transpose(0, 1)
        R = self.R_free[j].conj().transpose(0, 1)
        return W, R

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (B, 1, N)
        B, C, N = a.shape
        if C != self.IN_CHANNELS:
            raise RuntimeError(f"Expected {self.IN_CHANNELS} channels, got {C}")

        # v0 = P a
        v = torch.einsum("bcn,mc->bmn", a, self.P)  # (B, m, N)

        for k in range(self.N_LAYERS):
            W, R = self._layer_params(k)

            # W_k v
            v_lin = torch.einsum("ij,bjn->bin", W, v)  # (B, m, N)

            # Fourier part
            v_hat = torch.fft.rfft(v, dim=-1)          # (B, m, N//2+1) complex
            out_hat = torch.zeros_like(v_hat)

            modes = min(self.N_MODES, v_hat.shape[-1])
            v_hat_m = v_hat[..., :modes]               # (B, m, modes)
            R_m = R[..., :modes]                       # (m, m, modes)
            out_hat[..., :modes] = torch.einsum("ijm,bjm->bim", R_m, v_hat_m)

            v_fourier = torch.fft.irfft(out_hat, n=N, dim=-1)  # real (B, m, N)

            v = v_lin + v_fourier

        # G(a) = P^T v_L  (since P is real)
        out = torch.einsum("mc,bmn->bcn", self.P, v)  # (B, 1, N)
        return out


# ===========================
# Model with self-adjoint FNO blocks
# ===========================

class SymplecticFNO1d(torch.nn.Module):
    def __init__(self, modes=32, width=64, sa_layers=4):
        super().__init__()
        # Replace FNO1d blocks with self-adjoint linear FNO blocks
        self.fno0 = SelfAdjointLinearFNO1d(N_MODES=modes, WIDTH=width, N_LAYERS=sa_layers, IN_CHANNELS=1)
        self.fno1 = SelfAdjointLinearFNO1d(N_MODES=modes, WIDTH=width, N_LAYERS=sa_layers, IN_CHANNELS=1)
        self.fno2 = SelfAdjointLinearFNO1d(N_MODES=modes, WIDTH=width, N_LAYERS=sa_layers, IN_CHANNELS=1)
        self.fno3 = SelfAdjointLinearFNO1d(N_MODES=modes, WIDTH=width, N_LAYERS=sa_layers, IN_CHANNELS=1)
        self.fno4 = SelfAdjointLinearFNO1d(N_MODES=modes, WIDTH=width, N_LAYERS=sa_layers, IN_CHANNELS=1)

        self.mlp0 = create_mlp(INPUT_CHANNELS=1, OUTPUT_CHANNELS=1, HIDDEN_CHANNELS=[16, 16, 16])
        self.mlp1 = create_mlp(INPUT_CHANNELS=1, OUTPUT_CHANNELS=1, HIDDEN_CHANNELS=[16, 16, 16])
        self.mlp2 = create_mlp(INPUT_CHANNELS=1, OUTPUT_CHANNELS=1, HIDDEN_CHANNELS=[16, 16, 16])
        self.mlp3 = create_mlp(INPUT_CHANNELS=1, OUTPUT_CHANNELS=1, HIDDEN_CHANNELS=[16, 16, 16])
        self.mlp4 = create_mlp(INPUT_CHANNELS=1, OUTPUT_CHANNELS=1, HIDDEN_CHANNELS=[16, 16, 16])

    def forward(self, u):
        q, p = u[:, 0:1, :], u[:, 1:2, :]  # (B,1,N)
        # fixed grid (register once)
        x = torch.linspace(0.0, 1.0, 100, device=u.device).view(1, 1, 100)

        # M_up0
        tmp=self.fno0(p)
        tmp=self.mlp0(tmp)
        tmp=self.fno0(tmp)
        q = q + tmp*0.001

        # M_low1
        tmp=self.fno1(q)
        tmp=self.mlp1(tmp)
        tmp=self.fno1(tmp) 
        p = p + tmp*0.001

        # M_up2
        tmp=self.fno2(p)
        tmp=self.mlp2(tmp)
        tmp=self.fno2(tmp)
        q = q + tmp*0.001

        # M_low3
        tmp=self.fno3(q)
        tmp=self.mlp3(tmp)
        tmp=self.fno3(tmp)
        p = p + tmp*0.001

        # M_up4
        tmp=self.fno4(p)
        tmp=self.mlp4(tmp)
        tmp=self.fno4(tmp)
        q = q + tmp*0.001
        return torch.cat([q, p], dim=1)


def get_model(modes=32, width=84, sa_layers=4):
    return SymplecticFNO1d(modes=modes, width=width, sa_layers=sa_layers).to(device)


if __name__ == "__main__":
    inputs_train, targets_train, inputs_test, targets_test = load_wave_data("data/data.npz")

    BATCH_SIZE = 8
    EPOCHS = 500
    LEARNING_RATE = 0.001
    SAVE_INTERVAL = 5
    SAVE_DIR = "checkpoints"
    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(TensorDataset(inputs_test, targets_test), batch_size=BATCH_SIZE, shuffle=False)

    model = get_model()
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters:", total_params)

"""
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.MSELoss()

    start_time = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                test_loss += criterion(preds, y_batch).item()

        avg_test_loss = test_loss / len(test_loader)
        print(f"Epoch {epoch}/{EPOCHS}, Train Loss: {avg_loss:.10f}, Test Loss: {avg_test_loss:.10f}")

        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            elapsed = time.time() - start_time
            checkpoint_path = os.path.join(SAVE_DIR, f"sno_002_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            print(f"Elapsed: {elapsed:.2f}s ({elapsed/60:.2f} minutes)")

    print(f"Total training time: {(time.time() - start_time)/3600:.2f} hours")
"""