# Small-Scale LLM Training Framework Design

## Objective
- Deliver the project in exactly four gated milestones.
- Stop after each milestone for human review and explicit approval before any work on the next milestone.
- Make each milestone reviewable as a real GitHub delivery, not just as a partial internal implementation.

## Current Milestone 1 Status
- The repository contains a partial Milestone 1 implementation on `main`.
- Milestone 1 is not yet considered delivered.
- Remaining Milestone 1 work must close the gaps identified in human review:
  - a runnable end-to-end `train.py` for TinyStories on one GPU
  - a supported generation API for trained checkpoints
  - CI that runs a real training smoke and checks loss behavior
  - documentation written for external GitHub users with zero project context

## In Scope
- Tokenizer, model, training loop, optimizer, checkpointing
- Benchmarking, profiling, distributed and sharded training
- Common Crawl text extraction, filtering, and deduplication
- MATH evaluation, supervised fine-tuning, Expert Iteration, and GRPO

## Out of Scope
- UI, inference serving, or deployment platform work
- General MLOps platformization beyond milestone needs
- Product polish beyond what is needed to train, resume, and generate with the model
- Safety or alignment work beyond required filtering and verified-reward methods

## Assumptions
- Python and PyTorch remain the implementation stack.
- GitHub Actions remains the required CI system.
- TinyStories is the Milestone 1 primary training dataset for end-to-end delivery.
- OpenWebText remains part of the Milestone 1 data surface, but TinyStories is the required end-to-end user path.
- Milestone 1 targets a single GeForce RTX 3080 for GPU training and validation.
- External GitHub users must be able to understand setup, training, resume, and generation flows without prior project context.

## Open Questions For Human Review
- CI loss gate: use a deterministic TinyStories smoke fixture and require finite loss plus a bounded final loss after a fixed number of steps. The exact numeric upper bound should be locked during implementation after measuring the deterministic smoke run once.
- Generation API surface: library API is required; a separate CLI generator is optional and should not be added unless explicitly requested during task approval.

## Risks
- The Milestone 1 restriction on `torch.nn`, `torch.nn.functional`, and `torch.optim` still increases implementation and test complexity.
- A user-friendly `train.py` can broaden scope unless the feature set is kept to the minimum required single-GPU training path.
- CI training smoke must be small and deterministic enough for hosted runners while still proving real training behavior.
- Story fluency is qualitative; without a narrow acceptance path, review can drift into subjective redefinition of Milestone 1.

## Resolved Decisions
- GitHub is the source of truth for readiness and delivery state.
- Milestone 1 is not complete until a user can clone the repo, follow the docs, run a training entry point, resume from checkpoints, and generate story output from the trained model.
- Reviewer-facing milestone evidence and user-facing project documentation are both required; one does not replace the other.
- Milestone 1 recovery work must stay scoped to finishing the promised core training stack, not expanding into Milestone 2 features.

## Execution Model
- One program-level design doc governs the four-milestone sequence.
- Only the active milestone may have open implementation issues.
- Milestone 1 must treat the `torch.nn`, `torch.nn.functional`, and `torch.optim` restrictions as hard constraints, with only the explicitly permitted exceptions from `requirement.md`.
- Each implementation task must stay small and reviewable.
- Each milestone ends with:
  - user-facing documentation updated
  - reviewer evidence updated
  - validation artifacts collected
  - human review request sent
  - explicit stop on forward development
- No Milestone 2 or later work begins before written approval on the prior milestone.

## Milestone Plan

### Milestone 1: Core Training Stack
Goal:
- Deliver a real end-to-end small language model training project that an external GitHub user can install, train on TinyStories with one GPU, resume from checkpoints, and use to generate English story text.

Required Deliverables:
- BPE tokenizer
- Transformer language model
- cross-entropy loss
- AdamW optimizer
- training loop
- support for serializing and loading model state
- support for serializing and loading optimizer state
- runnable `train.py` for TinyStories single-GPU training
- supported generation API for trained checkpoints
- CI training smoke with loss validation
- GitHub-user-facing documentation for build, train, resume, and generation

Hard Constraints:
- Single-GPU support only.
- Do not use definitions from `torch.nn`, `torch.nn.functional`, or `torch.optim` other than the explicitly allowed exceptions in `requirement.md`.
- Keep the training or generation UX to the minimum needed for Milestone 1 delivery.

In Scope For Remaining Milestone 1 Work:
- Training entry point and config contract
- TinyStories end-to-end runnable training path
- Checkpoint-aware resume behavior in the public training flow
- Public generation API for trained checkpoints
- CI smoke training with loss assertions
- README and supporting docs written for external users

Out Of Scope For Remaining Milestone 1 Work:
- Multi-GPU training
- Benchmarking or profiling
- Inference service or web app
- Advanced sampling features beyond a minimal useful generation API
- New model architecture experiments

Task Breakdown:
1. Training UX and configuration surface
   - Add a runnable `train.py` entry point.
   - Define the minimum config and argument contract needed to train, resume, choose output paths, and point to TinyStories data.
   - Keep the training interface operational and reviewable rather than feature-rich.
2. TinyStories end-to-end training integration
   - Wire tokenizer training or loading, dataset preparation, batching, model build, optimizer build, training loop, checkpoint cadence, and resume behavior into the training entry point.
   - Prove the path works on the target single GPU.
3. Generation API for trained models
   - Expose a public Python API that loads the tokenizer and trained checkpoint, accepts a prompt, and returns generated English story text.
   - Keep the first version minimal and deterministic enough to test.
4. CI/CD upgrade with real training smoke
   - Add a deterministic TinyStories smoke path to CI that runs a few real training steps.
   - Assert training produces finite loss and that the final loss stays within an approved bound on the deterministic smoke fixture.
   - Keep the existing lint and unit checks.
5. GitHub-user-facing documentation
   - Rewrite the README and supporting docs for external users.
   - Cover install, dataset preparation, how to run `train.py`, how to resume, how to call the generation API, artifact layout, and expected outputs.
   - Keep reviewer-only milestone evidence separate from user onboarding.

Dependencies:
- Task 1 before Tasks 2, 3, and 5.
- Task 2 before Task 3 end-to-end validation.
- Task 2 before Task 4 final smoke configuration.
- Tasks 2, 3, and 4 before Task 5 is finalized.

Exit Evidence:
- `train.py` runs TinyStories training end to end on one GPU.
- Public training flow supports checkpoint save and resume.
- Public generation API loads the trained model and produces English story output from a prompt.
- CI runs a real training smoke and validates loss behavior automatically.
- README and supporting docs let a GitHub user build, train, resume, and generate without milestone-specific insider knowledge.
- Human review confirms Milestone 1 is delivered.

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
- Milestone 1 CI must gate merges with:
  - formatting and lint checks
  - unit tests
  - import or packaging sanity
  - deterministic training smoke that runs a few real training steps and checks loss behavior
- The training smoke may remain CPU-based if it is deterministic and reviewable, but the Milestone 1 exit gate still requires separate one-GPU validation on the RTX 3080.
- CI changes must stay minimal and milestone-specific; do not add unrelated platform work.

## Dependency and Installation Requirements
- Current required runtime dependency: PyTorch.
- No additional mandatory runtime dependency is planned for the remaining Milestone 1 recovery work unless a specific implementation task justifies it explicitly.
- Installation instructions must cover:
  - clean CPU setup
  - optional CUDA-enabled local environment for single-GPU training
  - TinyStories dataset acquisition or placement
  - how to run `train.py`
  - how to resume from checkpoints
  - how to call the generation API

## Documentation Requirements
- Root README must describe the project for external GitHub users with zero prior context.
- Documentation must explain:
  - what the project does
  - how to install it
  - how to prepare TinyStories
  - how to run end-to-end training
  - how to resume training
  - how to generate text from a trained checkpoint
  - what artifacts are created and where they live
- Reviewer-facing milestone runbooks may remain, but they must not substitute for user-facing docs.

## Proposed Remaining Milestone 1 Ticket Set After Human Approval
1. Runnable TinyStories training entry point
   - Objective: add `train.py`, argument/config parsing, output directory handling, and checkpoint-aware resume wiring.
2. End-to-end TinyStories training integration
   - Objective: connect tokenizer, dataset ingestion, model, optimizer, training loop, and checkpoint cadence into the public training path.
3. Trained-model generation API
   - Objective: add a public Python API that loads tokenizer plus checkpoint and generates story text from a prompt.
4. CI training smoke and loss gate
   - Objective: add deterministic smoke training in GitHub Actions and assert acceptable loss behavior.
5. GitHub-user documentation refresh
   - Objective: rewrite README and supporting docs around install, data prep, training, resume, generation, and artifact inspection.
