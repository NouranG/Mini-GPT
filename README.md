# Mini-GPT: Character-Level Language Model

A character-level Generative Pre-trained Transformer (GPT) implemented from scratch using PyTorch. This project explores the core components of autoregressive language modeling, from text preprocessing and next-character prediction to Transformer architecture, training, and text generation.

## Overview

Mini-GPT learns patterns from a text corpus and generates continuations from a user-provided prompt. Rather than relying on a pretrained language model, it implements the core GPT architecture and trains the model on a custom text dataset.

The project includes a configurable Transformer model, a training and validation pipeline, autoregressive text generation, and a simple Gradio interface for interactive inference.

## Key Features

- **Character-level tokenization:** Builds a vocabulary from unique characters in the training corpus.
- **Token and positional embeddings:** Represents characters and their positions in the input sequence.
- **Causal self-attention:** Prevents the model from attending to future characters during prediction.
- **Multi-head attention:** Learns contextual relationships through multiple attention heads.
- **Transformer blocks:** Combines attention, feed-forward networks, layer normalization, and residual connections.
- **Training and validation:** Optimizes next-character prediction using cross-entropy loss and AdamW.
- **Best-model checkpointing:** Saves model weights when validation loss improves.
- **Interactive inference:** Provides a Gradio interface for entering prompts and generating continuations.
- **Configurable hyperparameters:** Stores model, training, data, and generation settings in YAML.

## Architecture

The model follows a decoder-style Transformer architecture:

1. **Text preprocessing:** Reads a text corpus and creates character-to-index and index-to-character mappings.
2. **Sequence preparation:** Creates fixed-length input sequences and their one-character-shifted targets.
3. **Embedding layer:** Adds token embeddings and positional embeddings.
4. **Transformer blocks:** Applies causal multi-head self-attention and feed-forward transformations, with residual connections and layer normalization.
5. **Language modeling head:** Projects hidden representations into vocabulary-sized logits.
6. **Autoregressive generation:** Samples the next character from the model's predicted probability distribution and appends it to the sequence.

The model is trained to predict the next character at every position using cross-entropy loss.

## Technology Stack

- Python
- PyTorch
- NumPy
- Gradio
- PyYAML
- tqdm
- uv for Python dependency management

## Project Structure

```text
Mini-GPT/
├── Src/
│   ├── Embedding/
│   │   └── embedding.py
│   ├── Preprocessor/
│   │   ├── dataloader.py
│   │   ├── dataset.py
│   │   └── preprocess.py
│   ├── Tokenizer/
│   │   └── tokenize.py
│   ├── attention/
│   │   ├── multi_head_attention.py
│   │   └── self_attention.py
│   ├── inference/
│   │   └── generate.py
│   ├── model/
│   │   ├── gpt.py
│   │   └── transformer_block.py
│   └── training/
│       └── train.py
├── configs/
│   └── config.yaml
├── main.py
├── pyproject.toml
└── uv.lock
```

## Configuration

Model and training parameters are defined in `configs/config.yaml`.

| Parameter | Default value |
|---|---:|
| Embedding dimension | 256 |
| Attention heads | 8 |
| Transformer blocks | 6 |
| Context length | 64 characters |
| Batch size | 32 |
| Training epochs | 20 |
| Learning rate | 0.0003 |
| Weight decay | 0.01 |
| Training split | 90% |
| Validation split | 10% |
| Maximum new tokens | 200 |

These values can be modified to experiment with model capacity, context length, and training behavior.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/NouranG/Mini-GPT.git
cd Mini-GPT
```

### 2. Install dependencies

The project uses `uv` for dependency management.

```bash
uv sync
```

Alternatively, install the dependencies listed in `pyproject.toml` in a compatible Python environment.

### 3. Prepare the dataset

Create the `data/` directory and place the training corpus at:

```text
data/input.txt
```

The dataset is a plain-text file. The character vocabulary is built from the corpus, so the text should contain all characters expected in prompts during inference.

### 4. Train the model

Run the training module from the repository root:

```bash
uv run python -m Src.training.train
```

The training pipeline reports training and validation loss after each epoch and saves the weights of the best-performing checkpoint to:

```text
models/best_model.pt
```

Ensure that the `models/` directory exists before training.

### 5. Generate text from the command line

After training, run:

```bash
uv run python -m Src.inference.generate
```

Enter a prompt when requested. The model generates a continuation based on the learned character-level patterns.

### 6. Launch the Gradio interface

Run:

```bash
uv run python main.py
```

Open the local Gradio URL printed in the terminal, enter a prompt, and view the generated text.

**Note:** The dataset and trained model checkpoint are not included in the repository by default. Training must be completed successfully before inference can load the expected checkpoint.

## Training and Evaluation

The training process uses a next-character prediction objective.

- **Loss function:** Cross-entropy loss over the character vocabulary.
- **Optimizer:** AdamW.
- **Validation:** Evaluates average loss on a held-out portion of the encoded corpus.
- **Checkpoint selection:** Retains the model with the lowest observed validation loss.

Training and validation loss provide a useful initial assessment of learning and potential overfitting. Generated text should also be inspected qualitatively for coherence, repetition, and how well it continues the prompt.

## Learning Objectives

This project provides hands-on experience with:

- Understanding the internal components of GPT-style language models.
- Implementing scaled dot-product attention and causal masking.
- Building multi-head attention and Transformer blocks.
- Preparing sequential training examples for autoregressive prediction.
- Implementing a PyTorch training loop and validation workflow.
- Performing iterative text generation.
- Exposing a trained model through a lightweight interactive interface.

## Limitations

- The model operates at the **character level**, rather than using word-level or subword tokenization.
- Generation quality depends heavily on the training corpus, its size, and the number of training steps.
- The model is a small educational implementation, not a production-scale language model.
- The context window limits how much preceding text the model can consider at once.
- The current implementation uses sampling from the predicted probability distribution; advanced decoding strategies and generation controls can be added.

## Future Improvements

Potential extensions include:

- Adding configurable temperature and top-k/top-p sampling.
- Tracking and plotting training and validation loss.
- Adding random-seed control and reproducible experiments.
- Improving checkpoint management and model-loading safeguards.
- Adding automated tests for attention masking, tensor shapes, and generation.
- Exploring subword tokenization and larger training corpora.
- Comparing model configurations and evaluating generation quality systematically.

## Author

**Nouran Gamal**

GitHub: [@NouranG](https://github.com/NouranG)

---

This project is intended for educational purposes and for exploring the fundamentals of GPT-style Transformer language models.
