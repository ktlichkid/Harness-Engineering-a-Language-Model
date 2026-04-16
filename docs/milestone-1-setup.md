# Milestone 1 Setup

## Purpose
- Describe the current Milestone 1 repository state.
- Record setup steps that are valid today without guessing future commands.
- Mark the sections that must be updated as Milestone 1 implementation lands.

## Current Repository State
- The repository currently contains `requirement.md` and `AGENTS.md` on `main`.
- The approved program design exists, but it is not merged into `main` yet.
- Milestone 1 implementation code has not landed yet.
- No Python package scaffold, dependency manifest, CI workflow, or runnable trainer exists on `main` yet.

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
- There is no repository-supported install command yet.
- There are no repository-supported test, lint, or training commands yet.
- GPU setup instructions for the RTX 3080 are not documented yet because the Milestone 1 training stack has not landed.

## Required Follow-Up Updates
- Add Python environment and dependency installation steps once the scaffold and dependency choices are merged.
- Add local quality-check commands once test and CI wiring exist.
- Add dataset preparation commands once TinyStories and OpenWebText ingestion flows exist.
- Add training and checkpoint commands once the trainer and serialization paths exist.
- Add GPU-specific validation notes once the single-GPU integration path is implemented.

## Documentation Update Rule
- Update this document only with commands that exist in the repository at the time of the edit.
- If a Milestone 1 command is still pending, record it as a follow-up item instead of inventing syntax or behavior.
