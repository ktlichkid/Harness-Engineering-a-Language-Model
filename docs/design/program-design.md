# Small-Scale LLM Training Framework Design

## Objective
- Deliver the project in exactly four gated milestones.
- Stop after each milestone for human review and explicit approval before any work on the next milestone.
- Keep implementation traceable through small issues, isolated branches, CI checks, and milestone-specific documentation.

## In Scope
- Tokenizer, model, training loop, optimizer, checkpointing
- Benchmarking, profiling, distributed and sharded training
- Common Crawl text extraction, filtering, and deduplication
- MATH evaluation, supervised fine-tuning, Expert Iteration, and GRPO

## Out of Scope
- End-user productization
- UI, inference serving, or deployment platform work
- General MLOps platformization beyond milestone needs
- Safety or alignment work beyond required filtering and verified-reward methods

## Assumptions
- Python and PyTorch are the primary implementation stack.
- GitHub Actions is the default CI system unless directed otherwise.
- TinyStories, OpenWebText, Common Crawl, and MATH are acceptable upstream datasets for this project.
- Milestone 1 should be the only milestone ticketed immediately after approval; later milestones remain gated.

## Risks
- The Milestone 1 restriction on `torch.nn`, `torch.nn.functional`, and `torch.optim` increases implementation and test complexity.
- GPU-dependent validation may not be fully executable in hosted CI and may require a split between CPU CI and manual or self-hosted GPU validation.
- Flash Attention 2 Triton work and FSDP or sharding work can create deep coupling in Milestone 2 if task boundaries are not enforced.
- Data filtering quality and reasoning reward design can consume substantial time without clear acceptance metrics.

## Open Questions
- What GPU environment is available for Milestone 1 and later multi-GPU milestones?
- Should GitHub Actions be the required CI target, or is another CI system expected?
- What model size and training budget should define "runs successfully" for Milestone 1?
- Are there repository, branch, or reviewer conventions beyond the provided AGENTS instructions?
- What evidence will count as "demonstrated at the intended level" for Milestone 4?

## Execution Model
- One program-level design doc governs the four-milestone sequence.
- Only the active milestone may have open implementation issues.
- Each milestone ends with:
  - documentation updated
  - validation artifacts collected
  - human review request sent
  - explicit stop on forward development
- No Milestone 2 or later implementation begins before written approval on the prior milestone.

## Milestone Plan

### Milestone 1: Core Training Stack
Goal:
- Produce a single-GPU trainer with custom core components, checkpointing, automated tests, CI, and complete documentation.

Task Breakdown:
1. Repository scaffold and developer workflow
   - Define package layout, configuration layout, local quality checks, and CI entry points.
2. Data and tokenizer foundation
   - Implement BPE tokenizer training, encode/decode, dataset loading, and preprocessing for TinyStories and OpenWebText.
3. Core model and math primitives
   - Implement transformer blocks and cross-entropy without forbidden `torch.nn` or `torch.nn.functional` definitions.
4. Optimizer and training loop
   - Implement AdamW, batching, gradient flow, logging, checkpoint save/load, and resume behavior.
5. Validation and documentation
   - Add component and integration tests, CI coverage, setup instructions, architecture notes, and milestone runbook.

Dependencies:
- Task 1 before all others.
- Task 2 before full training integration.
- Task 3 before Task 4.
- Task 4 before Task 5 integration validation.

Exit Evidence:
- Single-GPU train run completes.
- Model and optimizer checkpoints save and reload correctly.
- Tests pass.
- CI runs successfully.
- Documentation is complete.

### Milestone 2: Performance and Multi-GPU Training
Goal:
- Add measurable performance infrastructure and operational multi-GPU training.

Task Breakdown:
1. Benchmark and profiling harness
2. Activation checkpointing integration
3. Flash Attention 2 Triton kernel
4. DDP training path
5. Optimizer sharding and FSDP path
6. Multi-GPU validation and docs

Dependencies:
- Milestone 1 approval is required before any Milestone 2 issue is opened.
- Harness work should precede optimization and distributed tuning.

### Milestone 3: Data Processing Pipeline
Goal:
- Turn raw Common Crawl inputs into filtered, deduplicated training-ready text.

Task Breakdown:
1. HTML ingestion and text extraction
2. Content filtering pipeline
3. PII and harmful-content filtering
4. Deduplication
5. Data output contract, validation, and docs

Dependencies:
- Milestone 2 approval is required before any Milestone 3 issue is opened.

### Milestone 4: Reasoning Evaluation and Post-Training Improvement
Goal:
- Add reasoning evaluation and verified-reward post-training methods.

Task Breakdown:
1. MATH zero-shot evaluation harness
2. Reasoning-trace SFT pipeline
3. Expert Iteration implementation
4. GRPO implementation
5. Verified reward evaluation, comparison, and docs

Dependencies:
- Milestone 3 approval is required before any Milestone 4 issue is opened.

## CI/CD Requirements
- Milestone 1 must add automated checks for linting, tests, and packaging or import sanity.
- CI must gate merges for milestone-scoped branches or PRs.
- GPU-required validation should be explicitly separated from CPU CI if hosted runners cannot execute it.
- Later milestones may extend CI, but milestone-specific validation must stay reviewable and reproducible.

## Dependency and Installation Requirements
- Expected baseline: Python, PyTorch, tokenizer and dataset-processing dependencies, test runner, lint or format tooling, and CI config.
- Exact dependency additions should be proposed during milestone ticketing, not assumed in this design doc.
- Installation instructions must cover local setup, dataset acquisition steps, and any GPU-specific prerequisites.

## Documentation Requirements
- Root README with project overview and milestone status
- Developer setup guide
- Milestone-specific runbooks for training, validation, and review evidence
- Architecture notes for major subsystem boundaries
- Dataset preparation instructions and provenance notes where relevant

## Proposed First Ticket Set After Approval
1. Create Milestone 1 repository scaffold, local quality workflow, and CI baseline.
2. Implement tokenizer and dataset preprocessing pipeline for TinyStories and OpenWebText.
3. Implement transformer core and custom loss path under Milestone 1 restrictions.
4. Implement AdamW, training loop, and checkpoint save/load.
5. Add Milestone 1 tests, integration validation, and documentation.

