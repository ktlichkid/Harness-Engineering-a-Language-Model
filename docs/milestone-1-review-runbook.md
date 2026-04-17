# Milestone 1 Review Runbook

## Purpose
- Define the evidence required for human approval of Milestone 1.
- Keep review expectations aligned with the approved design and the delivered `main` branch state.

## Milestone 1 Approval Gate
Human approval should only happen after the team presents reviewable evidence that Milestone 1 satisfies the approved design, the product requirements, and the milestone stop gate.

## Current Milestone 1 Status
- All Milestone 1 implementation issues are merged on `main`.
- Milestone 1 forward development is complete.
- Milestones 2-4 remain gated and must not begin until explicit human approval is given for Milestone 1.

## Required Evidence for Human Review
1. Implementation scope evidence
   - All Milestone 1 issues `#3` through `#20` are closed and traceable to their merged PRs.
   - The merged Milestone 1 code stays within the approved issue scopes.
2. Core training stack evidence
   - BPE tokenizer: `src/small_scale_llm/tokenizer/`
   - Transformer language model: `src/small_scale_llm/model/`
   - Custom cross-entropy loss: `src/small_scale_llm/model/loss.py`
   - AdamW optimizer: `src/small_scale_llm/optim/adamw.py`
   - Training loop: `src/small_scale_llm/training/step.py`
   - Model and optimizer checkpoint helpers: `src/small_scale_llm/checkpointing/`
   - The implementation respects the Milestone 1 restrictions on `torch.nn`, `torch.nn.functional`, and `torch.optim`, except for the documented allowed exceptions.
3. Validation evidence
   - Focused CPU tests are present under `tests/unit/`.
   - CPU-based GitHub Actions checks are defined in `.github/workflows/ci.yml`.
   - A single-GPU training run on the target RTX 3080 is implemented in `tests/integration/run_issue18_single_gpu.py`.
   - Model and optimizer checkpoint save or load behavior is demonstrated in the single-GPU artifacts and unit tests.
4. Output evidence
   - Review artifacts are committed under `artifacts/issue18/`:
     - `run_summary.json`
     - `training_log.json`
     - `generated_story.txt`
5. Documentation evidence
   - `README.md`, `docs/milestone-1-setup.md`, and this runbook match the delivered Milestone 1 implementation and evidence path.

## Reviewer Checklist
- Confirm the milestone only includes Milestone 1 work.
- Confirm all required Milestone 1 issues and PR links are traceable through the closed GitHub issue set.
- Confirm automated CPU checks are present and passing.
- Confirm the GPU run evidence is present in `artifacts/issue18/`.
- Confirm checkpoint evidence is present in `artifacts/issue18/run_summary.json` and the checkpoint unit tests.
- Confirm sample output evidence is present in `artifacts/issue18/generated_story.txt`.
- Confirm documentation matches the commands and workflows that exist in the repository.
- Confirm the team has stopped forward development pending approval.

## Evidence Packaging Guidance
- Prefer a single milestone approval note or review comment that links the exact proof for each checklist item.
- Use concise artifact names so a reviewer can map each artifact to one acceptance requirement quickly.
- If any required evidence is missing, do not request milestone approval yet.

## Suggested Approval Inputs
- Code and tests on `main`
- GitHub Actions `baseline-checks` history on the merged Milestone 1 PRs
- GPU evidence in `artifacts/issue18/`
- Setup and delivery docs in `README.md` and `docs/`

## Stop-Gate Reminder
Milestone 1 is ready for human review, but it is not approved yet. No Milestone 2 implementation should begin until a human explicitly accepts the Milestone 1 delivery package.
