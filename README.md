# Small-Scale LLM Training Framework

This repository contains the delivered Milestone 1 training stack for a small transformer language model built with explicit, reviewable PyTorch primitives.

It includes:
- deterministic BPE tokenizer training and runtime encode/decode
- TinyStories and OpenWebText dataset ingestion helpers
- transformer model assembly, custom loss, and AdamW optimizer
- training-step orchestration and model or optimizer checkpoint save/load
- a checked-in single-GPU integration harness that trains, resumes from checkpoint, and generates a sample story

Milestone 1 is implemented on `main`. Milestones 2-4 remain gated until Milestone 1 receives explicit human approval.

## What You Can Do Today
- Build the repository in a clean Python environment
- Run the CPU validation suite used by CI
- Train a tiny language model end to end from raw text using the public Python APIs
- Save and reload model or optimizer checkpoints
- Run the shipped CUDA integration harness to reproduce the Milestone 1 review evidence
- Generate a small text sample from the trained model

## Repository Layout
- `src/small_scale_llm/`: tokenizer, data, model, optimizer, training, and checkpointing code
- `tests/unit/`: CPU-executable unit coverage for component contracts
- `tests/integration/run_issue18_single_gpu.py`: end-to-end CUDA review harness
- `configs/milestone1/`: checked-in dataset configuration examples
- `artifacts/issue18/`: committed RTX 3080 run summary, logs, and generated sample
- `docs/`: setup guide, review runbook, and design docs

## Requirements
- Python 3.9+
- `pip`
- For CPU development: no extra system dependency beyond the normal PyTorch install
- For the shipped GPU review path: a CUDA-capable environment with a visible NVIDIA GPU

## Build and Install
Create a clean virtual environment and install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

## Validate the CPU Baseline
These are the repository-supported baseline checks and match the GitHub Actions CI surface:

```powershell
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m compileall src tests
.\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

## Train and Generate End to End in Python
The project does not yet ship a dedicated CLI trainer. The supported end-to-end training surface is the public Python API plus the checked-in integration harness.

The example below trains a tiny model on a small in-memory corpus, saves checkpoints, reloads them, and generates a sample continuation:

```python
from pathlib import Path

import torch

from small_scale_llm.checkpointing import (
    load_model_checkpoint,
    load_optimizer_checkpoint,
    save_model_checkpoint,
    save_optimizer_checkpoint,
)
from small_scale_llm.model import TransformerLanguageModel
from small_scale_llm.optim import AdamW
from small_scale_llm.tokenizer import BPETokenizer, train_bpe_from_texts, write_bpe_artifact
from small_scale_llm.training import run_training_step


def generate_text(model, tokenizer, prompt, *, max_sequence_length, max_new_tokens, device):
    generated = list(tokenizer.encode(prompt))
    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            window = generated[-max_sequence_length:]
            inputs = torch.tensor([window], dtype=torch.int64, device=device)
            logits = model(inputs)
            next_token = int(logits[0, -1].argmax().item())
            generated.append(next_token)

    return tokenizer.decode(generated)


texts = [
    "once upon a time there was a brave cat who loved warm soup",
    "once upon a time there was a kind fox who loved bright stars",
    "the brave cat found a tiny lantern and smiled at the moon",
    "the kind fox shared soft bread and told a happy story",
]

artifact = train_bpe_from_texts(texts, target_vocab_size=96)
tokenizer = BPETokenizer.from_artifact(artifact)
write_bpe_artifact(artifact, Path("artifacts") / "toy-bpe.json")

encoded = [tokenizer.encode(text) for text in texts]
max_sequence_length = max(len(token_ids) for token_ids in encoded) - 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerLanguageModel(
    vocab_size=len(tokenizer.vocab),
    max_sequence_length=max_sequence_length,
    hidden_size=64,
    num_heads=4,
    intermediate_size=128,
    num_layers=2,
).to(device)
optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)

batches = [torch.tensor([token_ids], dtype=torch.int64, device=device) for token_ids in encoded]
for step in range(40):
    log = run_training_step(model, optimizer, batches[step % len(batches)], step_index=step)
    if step % 10 == 0:
        print(log)

model_checkpoint = save_model_checkpoint(model, Path("artifacts") / "toy-model.pt")
optimizer_checkpoint = save_optimizer_checkpoint(optimizer, Path("artifacts") / "toy-optimizer.pt")

reloaded_model = TransformerLanguageModel(
    vocab_size=len(tokenizer.vocab),
    max_sequence_length=max_sequence_length,
    hidden_size=64,
    num_heads=4,
    intermediate_size=128,
    num_layers=2,
).to(device)
reloaded_optimizer = AdamW(reloaded_model.parameters(), lr=0.01, weight_decay=0.0)

load_model_checkpoint(reloaded_model, model_checkpoint)
load_optimizer_checkpoint(reloaded_optimizer, optimizer_checkpoint, map_location=device)

sample = generate_text(
    reloaded_model,
    tokenizer,
    "once upon a time there was",
    max_sequence_length=max_sequence_length,
    max_new_tokens=16,
    device=device,
)
print(sample)
```

What this flow does:
1. Trains a BPE tokenizer from raw text samples.
2. Builds a transformer language model sized to the trained vocabulary.
3. Runs autoregressive next-token training with the shipped training-step API.
4. Saves both model and optimizer checkpoints.
5. Reloads those checkpoints into fresh instances.
6. Uses greedy decoding to generate a text continuation.

## Run the Shipped Single-GPU Integration Harness
For the exact Milestone 1 review path, use the checked-in integration script. It trains on CUDA, validates checkpoint resume, and writes reproducible review artifacts.

First verify that your GPU-enabled Python environment can see CUDA:

```powershell
.\.venv-gpu\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

Then run the harness:

```powershell
.\.venv-gpu\Scripts\python tests\integration\run_issue18_single_gpu.py
```

Expected outputs:
- `artifacts/issue18/run_summary.json`
- `artifacts/issue18/training_log.json`
- `artifacts/issue18/generated_story.txt`

The committed review evidence currently shows:
- RTX 3080 visibility
- checkpoint-resume equivalence with `model_max_abs_diff = 0.0`
- matching reference and resumed final losses
- a minimal generated story sample

## Data and Tokenizer Surfaces
- TinyStories ingestion: `small_scale_llm.data.tinystories`
- OpenWebText ingestion: `small_scale_llm.data.openwebtext`
- BPE training and artifact writing: `small_scale_llm.tokenizer.train_bpe_from_texts`, `train_bpe_from_tinystories`, and `write_bpe_artifact`
- Runtime tokenizer loading: `small_scale_llm.tokenizer.BPETokenizer` and `load_bpe_tokenizer`

## Model, Training, and Checkpoint Surfaces
- Model assembly: `small_scale_llm.model.TransformerLanguageModel`
- Custom loss: `small_scale_llm.model.cross_entropy_loss`
- Optimizer: `small_scale_llm.optim.AdamW`
- Training helpers: `small_scale_llm.training.prepare_language_model_batch`, `run_training_step`, and `run_training_loop`
- Checkpoint helpers: `small_scale_llm.checkpointing.save_model_checkpoint`, `load_model_checkpoint`, `save_optimizer_checkpoint`, and `load_optimizer_checkpoint`

## Current Limitations
- No packaged train or generate CLI yet
- No distributed or multi-GPU training path in Milestone 1
- No production inference server or sampling stack beyond simple greedy decoding
- The checked-in generation example is intentionally tiny and overfit; it proves the end-to-end path, not production quality

## Documentation
- Setup guide: [docs/milestone-1-setup.md](docs/milestone-1-setup.md)
- Review runbook: [docs/milestone-1-review-runbook.md](docs/milestone-1-review-runbook.md)
- Program design: [docs/design/program-design.md](docs/design/program-design.md)
