import torch
import yaml

from Src.model.gpt import GPT
from Src.Preprocessor.preprocess import TextDataset

# -----------------------
# Load Config
# -----------------------

with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

processor = TextDataset(
    config["data"]["dataset_path"]
)

model = GPT(
    vocab_size=processor.vocab_size,
    embed_dim=config["model"]["embed_dim"],
    num_heads=config["model"]["num_heads"],
    block_size=config["model"]["block_size"],
    num_layers=config["model"]["num_layers"],
)

model.load_state_dict(
    torch.load(
        "models/best_model.pt",
        map_location=device,
    )
)

model.to(device)
model.eval()


def generate_text(prompt, max_new_tokens=100):

    encoded = torch.tensor(
        [processor.encode(prompt)],
        dtype=torch.long,
    ).to(device)

    generated = model.generate(
        encoded,
        max_new_tokens=max_new_tokens,
    )

    return processor.decode(
        generated[0].tolist()
    )


if __name__ == "__main__":

    prompt = input("Prompt: ")

    print(generate_text(prompt))