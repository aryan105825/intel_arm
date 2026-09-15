import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.vla_model import MultiModalVLAPolicy

def train(data_path: str, output_path: str, epochs: int, batch_size: int = 32, lr: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training VLA policy on device: {device}")

    raw_data = np.load(data_path)
    all_obs = raw_data["observations"] # [E, T, 78]
    all_acts = raw_data["actions"]     # [E, T, 14]

    chunk_size = 50
    X_list, Y_list = [], []

    for ep in range(len(all_obs)):
        obs = all_obs[ep]
        acts = all_acts[ep]
        for t in range(0, len(obs) - chunk_size, 5):
            X_list.append(obs[t])
            Y_list.append(acts[t:t+chunk_size])

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    Y = torch.tensor(np.array(Y_list), dtype=torch.float32)

    # Synthetic language embedding: "Set up dinner table"
    lang_emb = torch.ones((X.shape[0], 32), dtype=torch.float32) * 0.5

    dataset = TensorDataset(X, lang_emb, Y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiModalVLAPolicy(obs_dim=78, lang_dim=32, action_dim=14, chunk_size=50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_x, batch_l, batch_y in loader:
            batch_x, batch_l, batch_y = batch_x.to(device), batch_l.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x, batch_l)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs} | MSE Loss: {avg_loss:.6f}")

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"Successfully exported trained PyTorch policy to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demonstrations.npz")
    parser.add_argument("--output", default="models/vla_policy.pt")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs)