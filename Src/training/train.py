import torch
from torch.optim import AdamW
import tqdm

from Src.model.gpt import GPT
from Src.Preprocessor.dataset import GPTDataset
from Src.Preprocessor.preprocess import TextDataset
from Src.Preprocessor.dataloader import create_dataloader

import yaml

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# -------------------------
# Hyperparameters


embed_dim = config["model"]["embed_dim"]
num_heads = config["model"]["num_heads"]
block_size=config["model"]["block_size"]
num_layers=config["model"]["num_layers"]

epochs=config["training"]["epochs"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]

# -------------------------
# Dataset


processor = TextDataset(config["data"]["dataset_path"])

encoded = processor.encoded_corpus

split_idx = int(0.9 * len(encoded))

train_data = encoded[:split_idx]
val_data = encoded[split_idx:]
print(f"Total tokens: {len(encoded)}")
print(f"Train tokens: {len(train_data)}")
print(f"Validation tokens: {len(val_data)}")
print(f"Block size: {block_size}")

dataset_train = GPTDataset(
    train_data,
    block_size
)
dataset_val = GPTDataset(
    val_data,
    block_size
)

train_loader = create_dataloader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True
)
val_loader = create_dataloader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False
)
print(len(dataset_train))
print(len(dataset_val))
# -------------------------
# Model



model = GPT(
    vocab_size=processor.vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=num_layers,
)

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)

# -------------------------
# evaluation pipeline

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)

best_val_loss = float("inf")


@torch.no_grad()
def evaluate(model, dataloader, device):

    model.eval()

    total_loss = 0

    for x, y in dataloader:

        x = x.to(device)
        y = y.to(device)

        _, loss = model(x, y)

        total_loss += loss.item()

    return total_loss / len(dataloader)

# -------------------------
# Training Loop

for epoch in range(epochs):

    model.train()

    total_loss = 0
    progress_bar = tqdm.tqdm(
    train_loader,
    desc=f"Epoch {epoch+1}/{epochs}",
    leave=True
)

    for x, y in progress_bar:

        x = x.to(device)
        y = y.to(device)

        logits, loss = model(x, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix(
    loss=f"{loss.item():.4f}"
)

    train_loss = total_loss / len(train_loader)

    val_loss = evaluate(
        model,
        val_loader,
        device
    )

    print(
        f"Epoch {epoch+1}/{epochs}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    #saving best model    
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "models/best_model.pt"
        )

        print(
            f"Saved best model "
            f"(Validation Loss: {val_loss:.4f})"
        )