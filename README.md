# Small-Scale LLM Training Framework

This repository ships the Milestone 1 training stack for a small language model built from custom tokenizer, model, loss, optimizer, training, and checkpoint components.

The public surface on `main` is:
- `train.py` for TinyStories training and resume
- `small_scale_llm.generation` for loading a trained checkpoint and generating story text
- CPU CI and smoke-validation workflows that exercise the shipped training path

## Repository Layout
- `train.py`: public TinyStories training entry point
- `src/small_scale_llm/`: tokenizer, model, optimizer, training, checkpointing, and generation APIs
- `configs/milestone1/`: checked-in TinyStories and training configs
- `tests/unit/`: CPU-executable unit coverage
- `tests/integration/run_issue48_ci_smoke.py`: deterministic smoke training path used in CI
- `tests/integration/run_issue18_single_gpu.py`: single-GPU review harness
- `artifacts/issue18/`: committed RTX 3080 review evidence
- `docs/`: setup, artifact, and review guidance

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

## Prepare TinyStories
The default training config is `configs/milestone1/train_tinystories.json`. It points at `configs/milestone1/tinystories.json`, targets `cuda`, downloads TinyStories automatically when the split file is missing, and writes outputs to `artifacts/tinystories-train`.

You have three supported data paths:
- Let `train.py` download `TinyStories-train.txt` automatically into `data/tinystories/`
- Place `TinyStories-train.txt` yourself at `data/tinystories/TinyStories-train.txt`
- Pass a different local file with `--tinystories-path`

## Train
Run the default single-GPU training path:

```powershell
.\.venv\Scripts\python train.py
```

Common overrides:

```powershell
.\.venv\Scripts\python train.py --output-dir artifacts/my-run
.\.venv\Scripts\python train.py --tinystories-path C:\data\TinyStories-train.txt --no-download-data
.\.venv\Scripts\python train.py --device cpu --output-dir artifacts/cpu-smoke
```

The CLI contract is:

```powershell
.\.venv\Scripts\python train.py --help
```

## Resume
Resume always reads the latest checkpoint recorded in `output_dir/training_state.json`:

```powershell
.\.venv\Scripts\python train.py --resume
.\.venv\Scripts\python train.py --resume --output-dir artifacts/my-run
```

## Generate Text
Load the most recent trained checkpoint from an output directory:

```powershell
.\.venv\Scripts\python -c "from small_scale_llm.generation import generate_story; print(generate_story('artifacts/tinystories-train', 'tiny story', device='cuda', max_new_tokens=32))"
```

Load a specific checkpoint instead of the latest recorded one:

```powershell
.\.venv\Scripts\python -c "from small_scale_llm.generation import generate_story; print(generate_story('artifacts/tinystories-train', 'tiny story', checkpoint_path='artifacts/tinystories-train/checkpoints/model-step-80.pt', device='cuda', max_new_tokens=32))"
```

For reusable API access:

```python
from small_scale_llm.generation import load_story_generator

generator = load_story_generator("artifacts/tinystories-train", device="cuda")
print(generator.generate("tiny story", max_new_tokens=32))
```

Use prompt text that is compatible with the tokenizer built during training. If you train on a very narrow local fixture, unseen characters or words can raise a tokenizer error during generation.

## Output Layout
Each training run writes:
- `resolved_train_config.json`: the fully resolved training contract for that run
- `training_state.json`: the current tokenizer path plus the latest model and optimizer checkpoints
- `tokenizer.json`: runtime tokenizer state used for training and generation
- `tokenizer_artifact.json`: serialized BPE training artifact
- `checkpoints/model-step-<N>.pt`: model checkpoints
- `checkpoints/optimizer-step-<N>.pt`: optimizer checkpoints

## CI And Review Evidence
The public validation surfaces are:
- `.github/workflows/ci.yml`: baseline formatting, lint, compile, and test checks on every PR and `main`
- `tests/integration/run_issue48_ci_smoke.py`: deterministic CPU smoke training with a bounded loss gate
- `tests/integration/run_issue18_single_gpu.py`: RTX 3080 end-to-end review harness
- `artifacts/issue18/run_summary.json`: GPU run summary and resume evidence
- `artifacts/issue18/training_log.json`: GPU training logs
- `artifacts/issue18/generated_story.txt`: GPU sample output

## Documentation
- External-user setup and workflow guide: [docs/milestone-1-setup.md](docs/milestone-1-setup.md)
- Milestone review runbook: [docs/milestone-1-review-runbook.md](docs/milestone-1-review-runbook.md)
- Program design: [docs/design/program-design.md](docs/design/program-design.md)
