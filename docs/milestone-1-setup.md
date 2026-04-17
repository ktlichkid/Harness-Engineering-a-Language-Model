# Milestone 1 Setup

## Purpose
- Describe the delivered Milestone 1 repository state on `main`.
- Record the repository-supported setup and validation steps that exist today.
- Separate CPU-baseline validation from the GPU-only integration evidence path.

## Current Repository State
- `src/small_scale_llm/` contains the delivered Milestone 1 tokenizer, dataset loaders, model stack, optimizer, training loop, and checkpoint helpers.
- `tests/unit/` contains focused CPU-executable coverage for the Milestone 1 component contracts.
- `tests/integration/run_issue18_single_gpu.py` contains the minimum RTX 3080 integration harness used for Milestone 1 review evidence.
- `artifacts/issue18/` contains committed review artifacts from the single-GPU integration run.
- `.github/workflows/ci.yml` contains the CPU-based baseline checks used to validate the Milestone 1 codebase on GitHub Actions.

## Review Prerequisites
1. Clone the repository.
2. Read `requirement.md` for the product requirements.
3. Read `docs/design/program-design.md` for the approved milestone plan.
4. Read `docs/milestone-1-review-runbook.md` before requesting human approval.

Example:

```powershell
git clone git@github.com:ktlichkid/Harness-Engineering-a-Language-Model.git
cd Harness-Engineering-a-Language-Model
```

## CPU Baseline Setup
The repository-supported CPU install path is a clean virtual environment plus the editable dev install:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Run the repository baseline checks from that environment:

```powershell
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m compileall src tests
.\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

These commands match the CPU-based baseline enforced in `.github/workflows/ci.yml`.

## Dataset and Training Surfaces
- TinyStories ingestion lives in `small_scale_llm.data.tinystories`.
- OpenWebText ingestion lives in `small_scale_llm.data.openwebtext`.
- BPE training and runtime tokenization live in `small_scale_llm.tokenizer`.
- Model, loss, optimizer, training, and checkpointing live under `small_scale_llm.model`, `small_scale_llm.optim`, `small_scale_llm.training`, and `small_scale_llm.checkpointing`.
- Milestone 1 does not include a dedicated end-user CLI; the supported evidence path is the checked-in integration harness plus the unit and CI validations above.

## Single-GPU Validation Path
Milestone 1 GPU validation was executed from an isolated `.venv-gpu` environment that is intentionally not part of the committed repository changes.

Use that isolated environment only for the single-GPU evidence path:

```powershell
.\.venv-gpu\Scripts\python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
.\.venv-gpu\Scripts\python tests\integration\run_issue18_single_gpu.py
```

Expected review artifacts:
- `artifacts/issue18/run_summary.json`
- `artifacts/issue18/training_log.json`
- `artifacts/issue18/generated_story.txt`

The exact CUDA-enabled PyTorch wheel source is environment-specific and intentionally was not committed into the repository setup path. The required condition for GPU validation is that the isolated environment reports the local RTX 3080 through `torch.cuda`.
