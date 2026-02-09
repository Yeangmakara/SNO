import numpy as np
import torch
from neuralop.models import FNO1d
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os
import time
# Automatically select GPU if available, otherwise use CPU
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

def get_fno_model(in_channels=2, out_channels=2, modes=32, width=160):
    model = FNO1d(
        n_modes_height=modes,
        hidden_channels=width,
        in_channels=in_channels,
        out_channels=out_channels
    )
    return model.to(device)  # Move model to GPU or CPU
if __name__=="__main__":
    inputs_train, targets_train, inputs_test, targets_test = load_wave_data(
        "data/data.npz"
    )

    batch_size = 8
    epochs = 100
    learning_rate = 0.001

    save_interval = 5
    save_dir = "checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(TensorDataset(inputs_train, targets_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(inputs_test, targets_test), batch_size=batch_size, shuffle=False)

    model = get_fno_model()
    # Calculate the total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print("Total Parameters: ",total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    # Start timing here clearly
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)  # Move data to GPU or CPU

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)

        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                preds = model(x_batch)
                loss = criterion(preds, y_batch)
                test_loss += loss.item()
        avg_test_loss = test_loss / len(test_loader)

        print(f"Epoch {epoch}/{epochs}, Train Loss: {avg_loss:.10f}, Test Loss: {avg_test_loss:.10f}")

        if epoch % save_interval == 0 or epoch == epochs:
            current_time = time.time()
            elapsed_time = current_time - start_time
            checkpoint_path = os.path.join(save_dir, f"fno2_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}: {checkpoint_path}")
            print(f"Elapsed time after epoch {epoch}: {elapsed_time:.2f}s ({elapsed_time/60:.2f} minutes)")

    total_time = time.time() - start_time
    print(f"Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    print("Training complete")
