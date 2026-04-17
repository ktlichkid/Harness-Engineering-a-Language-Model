# Small-Scale LLM Training Framework

Milestone 1 is implemented on `main` and is ready for human delivery review.
Milestones 2-4 remain gated and must not begin until Milestone 1 receives explicit human approval.

## Milestone 1 Status
- Complete core tokenizer, model, loss, optimizer, training, and checkpoint stack under `src/small_scale_llm/`
- CPU validation and CI baseline in `.github/workflows/ci.yml`
- Focused unit coverage under `tests/unit/`
- Single-GPU integration evidence under `tests/integration/` and `artifacts/issue18/`

## Repository Layout
- `src/small_scale_llm/`: Milestone 1 implementation
- `tests/unit/`: CPU-executable component tests
- `tests/integration/`: single-GPU integration harness
- `configs/`: checked-in dataset and milestone configuration
- `artifacts/issue18/`: committed review evidence from the RTX 3080 run
- `docs/`: design, setup, and review runbook documentation

## CPU Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
.\.venv\Scripts\python -m ruff format --check .
.\.venv\Scripts\python -m ruff check .
.\.venv\Scripts\python -m compileall src tests
.\.venv\Scripts\python -m unittest discover -s tests -p "test_*.py"
```

## Single-GPU Review Evidence
- Integration harness: `tests/integration/run_issue18_single_gpu.py`
- Summary artifact: `artifacts/issue18/run_summary.json`
- Training log artifact: `artifacts/issue18/training_log.json`
- Generated sample: `artifacts/issue18/generated_story.txt`

The committed evidence shows:
- RTX 3080 visibility
- model and optimizer checkpoint resume equivalence
- a minimal story-generation sample from the end-to-end run

## Documentation
- Setup guide: [docs/milestone-1-setup.md](docs/milestone-1-setup.md)
- Review runbook: [docs/milestone-1-review-runbook.md](docs/milestone-1-review-runbook.md)
- Program design: [docs/design/program-design.md](docs/design/program-design.md)
