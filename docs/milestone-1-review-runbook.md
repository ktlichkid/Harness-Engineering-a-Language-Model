# Milestone 1 Review Runbook

## Purpose
- Define the evidence required for Milestone 1 human approval.
- Keep reviewer expectations aligned with the shipped public training, resume, generation, CI, and artifact surfaces on `main`.
- Keep reviewer-only evidence separate from the external-user setup guide.

## Current Milestone 1 Status
- All Milestone 1 implementation work is merged on `main`.
- The delivery-recovery issues `#45` through `#49` are complete.
- Milestones 2-4 remain gated until human approval is given for Milestone 1.

## Required Evidence For Human Review
1. Public training and resume surface
   - `train.py` exists on `main`.
   - `configs/milestone1/train_tinystories.json` defines the public training contract.
   - `src/small_scale_llm/training/entrypoint.py` supports TinyStories training and resume from `training_state.json`.
2. Public generation surface
   - `src/small_scale_llm/generation/api.py` loads tokenizer and checkpoint artifacts from a completed training output directory.
   - The latest-checkpoint path and explicit-checkpoint override are both covered by tests.
3. Core training stack evidence
   - Tokenizer: `src/small_scale_llm/tokenizer/`
   - Transformer model and loss: `src/small_scale_llm/model/`
   - Optimizer: `src/small_scale_llm/optim/adamw.py`
   - Checkpoint helpers: `src/small_scale_llm/checkpointing/`
   - Training step logic: `src/small_scale_llm/training/step.py`
4. Automated validation evidence
   - GitHub baseline checks are defined in `.github/workflows/ci.yml`.
   - The deterministic smoke path is `tests/integration/run_issue48_ci_smoke.py`.
   - Focused unit tests exist under `tests/unit/`, including training and generation coverage.
5. Single-GPU review evidence
   - The RTX 3080 harness is `tests/integration/run_issue18_single_gpu.py`.
   - The committed review artifacts are:
     - `artifacts/issue18/run_summary.json`
     - `artifacts/issue18/training_log.json`
     - `artifacts/issue18/generated_story.txt`
6. Documentation evidence
   - `README.md` describes install, TinyStories setup, `train.py`, resume, generation, output layout, and CI surfaces for external users.
   - `docs/milestone-1-setup.md` matches the shipped public workflows.
   - This runbook matches the current review evidence package.

## Reviewer Checklist
- Confirm the public training path can be followed from `README.md` and `docs/milestone-1-setup.md` without milestone-internal context.
- Confirm `train.py` supports both first-run training and `--resume`.
- Confirm the generation API is present and documented.
- Confirm CI includes both the repository baseline checks and the deterministic training smoke.
- Confirm the GPU review artifacts exist under `artifacts/issue18/`.
- Confirm the sample output evidence exists in `artifacts/issue18/generated_story.txt`.
- Confirm merged GitHub issues and PRs are traceable for the Milestone 1 work, including the delivery-recovery issues `#45` through `#49`.
- Confirm Milestone 2 work has not started.

## Evidence Packaging Guidance
- Use one approval note or review comment that links the exact code, CI runs, and artifact files for each checklist item.
- Prefer direct links to merged PRs for issues `#45` through `#49` when presenting the recovery work.
- If any public-surface docs drift from the shipped commands or file layout, correct the docs before requesting approval.

## Suggested Approval Inputs
- `main` branch code and tests
- `.github/workflows/ci.yml` and recent green GitHub Actions runs
- `tests/integration/run_issue48_ci_smoke.py`
- `tests/integration/run_issue18_single_gpu.py`
- `artifacts/issue18/`
- `README.md` and `docs/milestone-1-setup.md`

## Stop-Gate Reminder
Milestone 1 is ready for human review but not yet human-approved. No Milestone 2 implementation should begin until a human explicitly accepts the Milestone 1 delivery package.
