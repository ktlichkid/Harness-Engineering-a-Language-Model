# Milestone 1 Review Runbook

## Purpose
- Define the evidence required for human approval of Milestone 1.
- Keep review expectations aligned with the approved design and current repository state.

## Milestone 1 Approval Gate
Human approval should only happen after the team presents reviewable evidence that Milestone 1 satisfies the approved design, the product requirements, and the milestone stop gate.

## Required Evidence for Human Review
1. Implementation scope evidence
   - PRs for Milestone 1 link to the relevant GitHub issues.
   - The merged Milestone 1 code stays within the approved issue scopes.
2. Core training stack evidence
   - BPE tokenizer, transformer language model, custom cross-entropy loss, AdamW optimizer, training loop, and checkpoint save or load paths are present.
   - The implementation respects the Milestone 1 restrictions on `torch.nn`, `torch.nn.functional`, and `torch.optim`, except for the documented allowed exceptions.
3. Validation evidence
   - Automated tests for implemented Milestone 1 components are present and passing.
   - CPU-based GitHub Actions checks pass for the Milestone 1 scope.
   - A single-GPU training run on the target RTX 3080 completes successfully.
   - Model and optimizer checkpoint save or load behavior is demonstrated.
4. Output evidence
   - Review artifacts include sample outputs showing at least basic fluency for simple children's story generation.
5. Documentation evidence
   - Setup, validation, and review documentation for Milestone 1 is updated to match the implementation that actually landed.

## Reviewer Checklist
- Confirm the milestone only includes Milestone 1 work.
- Confirm all required Milestone 1 issues and PR links are traceable.
- Confirm automated checks are attached or linked.
- Confirm the GPU run evidence is attached or linked.
- Confirm checkpoint evidence is attached or linked.
- Confirm sample output evidence is attached or linked.
- Confirm documentation matches the commands and workflows that exist in the repository.
- Confirm the team has stopped forward development pending approval.

## Evidence Packaging Guidance
- Prefer a single milestone review comment or PR summary that links the exact proof for each checklist item.
- Use concise artifact names so a reviewer can map each artifact to one acceptance requirement quickly.
- If any required evidence is missing, do not request milestone approval yet.

## Current State Note
- This runbook defines the approval target for Milestone 1.
- The current `main` branch does not yet contain the implementation, CI, or training artifacts listed above.
- As Milestone 1 issues land, update this runbook to replace placeholders with direct links to the actual evidence locations.
