import torch
from torch.optim import AdamW

from model.gpt import GPT
from Preprocessor.dataset import GPTDataset
from Preprocessor.dataloader import create_dataloader

# -------------------------
# Hyperparameters
# -------------------------

batch_size = 32
block_size = 128
embed_dim = 256
num_heads = 8
num_layers = 6
learning_rate = 3e-4
epochs = 10

# -------------------------
# Dataset
# -------------------------

dataset = GPTDataset(...)

train_loader = create_dataloader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# -------------------------
# Model
# -------------------------

model = GPT(
    vocab_size=dataset.vocab_size,
    embed_dim=embed_dim,
    num_heads=num_heads,
    block_size=block_size,
    num_layers=num_layers
)

optimizer = AdamW(
    model.parameters(),
    lr=learning_rate
)

# -------------------------
# Training Loop
# -------------------------

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for x, y in train_loader:

        logits, loss = model(x, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch+1}:",
        total_loss / len(train_loader)
    )