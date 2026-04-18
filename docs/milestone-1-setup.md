# Milestone 1 Setup And Workflow

## Purpose
- Help an external GitHub user install the repository with zero prior context.
- Document how to prepare TinyStories, run `train.py`, resume training, use the generation API, and inspect outputs.
- Separate everyday user workflows from milestone-review-only evidence.

## What Ships On `main`
- `train.py` trains or resumes the TinyStories model from the checked-in config contract.
- `small_scale_llm.generation` loads tokenizer and checkpoint artifacts from a completed training output directory.
- `.github/workflows/ci.yml` runs the repository baseline checks plus a deterministic training smoke.
- `artifacts/issue18/` contains the committed RTX 3080 review evidence used for Milestone 1 delivery review.

## Install
Create a clean environment and install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Optional repository baseline checks:

```powershell
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m compileall src tests
.\.venv\Scripts\python -m pytest
```

## TinyStories Data
The checked-in dataset config is `configs/milestone1/tinystories.json`.

Supported ways to provide TinyStories:
1. Let `train.py` download `TinyStories-train.txt` automatically into `data/tinystories/`.
2. Place the official train split yourself at `data/tinystories/TinyStories-train.txt`.
3. Point `train.py` at another local copy with `--tinystories-path`.

The default public training config is `configs/milestone1/train_tinystories.json`:
- device: `cuda`
- output directory: `artifacts/tinystories-train`
- total steps: `160`
- checkpoint interval: `80`

Inspect the CLI before running:

```powershell
.\.venv\Scripts\python train.py --help
```

## Run Training
Default single-GPU path:

```powershell
.\.venv\Scripts\python train.py
```

Use a different output directory:

```powershell
.\.venv\Scripts\python train.py --output-dir artifacts/my-run
```

Use a pre-downloaded TinyStories file and skip download:

```powershell
.\.venv\Scripts\python train.py --tinystories-path C:\data\TinyStories-train.txt --no-download-data
```

If you want a small CPU-only local verification path, point the public entry point at a CPU config or run the deterministic smoke harness described below. The checked-in default config is intended for the single-GPU path.

## Resume Training
Resume uses the latest checkpoint recorded in `training_state.json` inside the chosen output directory:

```powershell
.\.venv\Scripts\python train.py --resume
.\.venv\Scripts\python train.py --resume --output-dir artifacts/my-run
```

The resume contract depends on these files remaining together:
- `training_state.json`
- `tokenizer.json`
- `checkpoints/model-step-<N>.pt`
- `checkpoints/optimizer-step-<N>.pt`

## Generate Text From A Trained Run
Generate from the latest recorded checkpoint:

```powershell
.\.venv\Scripts\python -c "from small_scale_llm.generation import generate_story; print(generate_story('artifacts/tinystories-train', 'tiny story', device='cuda', max_new_tokens=32))"
```

Generate from an explicit checkpoint:

```powershell
.\.venv\Scripts\python -c "from small_scale_llm.generation import generate_story; print(generate_story('artifacts/tinystories-train', 'tiny story', checkpoint_path='artifacts/tinystories-train/checkpoints/model-step-80.pt', device='cuda', max_new_tokens=32))"
```

If you want to keep the loaded model in memory:

```python
from small_scale_llm.generation import load_story_generator

generator = load_story_generator("artifacts/tinystories-train", device="cuda")
print(generator.generate("tiny story", max_new_tokens=32))
```

The generation API can only encode text supported by the tokenizer saved in the training output directory. If you train on a narrow local fixture, use prompts that stay close to that fixture vocabulary.

## Output Files To Inspect
After a successful run, inspect:
- `artifacts/tinystories-train/resolved_train_config.json`
- `artifacts/tinystories-train/training_state.json`
- `artifacts/tinystories-train/tokenizer.json`
- `artifacts/tinystories-train/tokenizer_artifact.json`
- `artifacts/tinystories-train/checkpoints/`

These files are the supported bridge between training and generation.

## CI And Review Evidence
For repository validation and evidence:
- GitHub CI runs `.github/workflows/ci.yml`.
- The deterministic smoke path is `tests/integration/run_issue48_ci_smoke.py`.
- The committed single-GPU review harness is `tests/integration/run_issue18_single_gpu.py`.
- Review artifacts are under `artifacts/issue18/`.

Run the same deterministic smoke path locally with:

```powershell
.\.venv\Scripts\python tests\integration\run_issue48_ci_smoke.py
```

That script is the smallest shipped end-to-end training workflow in the repository and matches the CI smoke gate.

## Review-Only Material
If you are reviewing Milestone 1 delivery rather than just using the project:
- read [docs/milestone-1-review-runbook.md](docs/milestone-1-review-runbook.md)
- inspect `artifacts/issue18/`
- inspect the GitHub Actions history for the merged Milestone 1 PRs
