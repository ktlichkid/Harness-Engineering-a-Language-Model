# Milestone 1 Setup

## Purpose
- Describe the current Milestone 1 repository state.
- Record setup steps that are valid today without guessing future commands.
- Mark the sections that must be updated as Milestone 1 implementation lands.

## Current Repository State
- The repository contains the Milestone 1 package scaffold under `src/`, `tests/`, and `configs/`.
- The repository contains the Milestone 1 setup and review docs under `docs/`.
- No runnable trainer, dataset ingestion flow, tokenizer, model, optimizer, or checkpoint implementation exists on `main` yet.
- This issue adds the initial CPU-based local quality and CI baseline.

## Current Review Prerequisites
1. Clone the repository.
2. Read `requirement.md` for the product requirements.
3. If needed for planning context, review the approved design doc from its open or merged PR state rather than assuming it exists on `main`.

Example:

```bash
git clone git@github.com:ktlichkid/Harness-Engineering-a-Language-Model.git
cd Harness-Engineering-a-Language-Model
```

## Current Setup Boundary
- The repository-supported install path for baseline quality checks is a clean virtual environment plus `pip install -e ".[dev]"`.
- The repository-supported quality commands are limited to formatting, linting, compile checks, and the current test suite.
- GPU setup instructions for the RTX 3080 are not documented yet because the Milestone 1 training stack has not landed.

## Local Quality Baseline
Run the baseline quality checks from a clean virtual environment:

```bash
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m compileall src tests
.venv\Scripts\python -m pytest
```

These commands match the CPU-based baseline configured in GitHub Actions and avoid relying on unrelated global Python packages in a developer environment.

## Required Follow-Up Updates
- Add Python environment and dependency installation steps once the scaffold and dependency choices are merged.
- Add dataset preparation commands once TinyStories and OpenWebText ingestion flows exist.
- Add training and checkpoint commands once the trainer and serialization paths exist.
- Add GPU-specific validation notes once the single-GPU integration path is implemented.

## Documentation Update Rule
- Update this document only with commands that exist in the repository at the time of the edit.
- If a Milestone 1 command is still pending, record it as a follow-up item instead of inventing syntax or behavior.
