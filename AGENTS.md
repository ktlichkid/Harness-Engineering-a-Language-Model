# AGENTS.md

## Purpose

This file defines universal development-process rules for all agents working in this repository.

These rules are intentionally:
- role-neutral
- project-neutral
- implementation-neutral

All agents must follow these rules unless a stricter instruction is explicitly provided elsewhere.

---

## Core Operating Principles

1. **Work must be traceable.**
   - Every meaningful change must map to a tracked task, issue, or explicit request.
   - Work should be easy to audit after the fact.

2. **Small, reviewable changes are preferred.**
   - Favor focused changes over large, sweeping edits.
   - Avoid mixing unrelated modifications in one unit of work.

3. **Do not act on assumptions when uncertainty is important.**
   - If a missing detail could materially affect correctness, scope, security, or behavior, surface the uncertainty clearly.
   - Prefer the narrowest safe interpretation over speculative expansion.

4. **Do not silently broaden scope.**
   - Only perform the work that is required.
   - Do not add adjacent fixes, refactors, features, or cleanup unless explicitly requested.

5. **Preserve repository integrity.**
   - Protect stability, reviewability, reproducibility, and auditability at all times.

---

## Task Discipline

### Start Conditions
Before beginning implementation work:
- confirm the work item being addressed
- identify the relevant files or areas
- read existing local patterns before editing
- prefer understanding before modification

### Scope Rules
Agents must:
- stay within the assigned or clearly implied scope
- avoid unrelated edits
- avoid opportunistic refactors
- avoid introducing behavior changes outside the requested work

Agents must not:
- self-expand the task
- bundle unrelated improvements into the same change
- reinterpret vague ideas as approved requirements

### Completion Discipline
Before considering work complete:
- ensure the requested change has been made
- ensure unrelated areas were not modified without reason
- run relevant validation where applicable
- report honestly what was and was not verified

---

## Design Doc Requirements

1. **Any non-trivial work must have a design doc before implementation begins.**
   - If the work involves multiple files, multiple steps, shared interfaces, workflow changes, CI/CD changes, dependency changes, documentation impact, or non-obvious design choices, a design doc is required.

2. **A design doc must include task breakdown.**
   - The design doc must break the work into concrete, reviewable tasks.
   - Task boundaries must be explicit enough that implementation can be assigned and tracked without ambiguity.

3. **A design doc must include CI/CD requirements.**
   - The design doc must state any required CI/CD changes, validation expectations, checks, gates, deployment workflow implications, or explicit statement that no CI/CD changes are required.

4. **A design doc must include dependency and installation requirements.**
   - The design doc must state whether new dependencies are required.
   - The design doc must state any install, setup, bootstrap, migration, or environment preparation requirements.
   - If no dependency or installation changes are needed, that should be stated explicitly.

5. **A design doc must include documentation requirements.**
   - The design doc must state which documentation must be created, updated, or confirmed unchanged.
   - This includes developer documentation, setup instructions, operational notes, and user-facing documentation where relevant.

6. **A design doc must receive final human review before implementation is authorized.**
   - Agent review may assist, but it does not replace final human review.
   - Implementation must not begin until the design doc has been reviewed and approved by a human.

7. **Design docs must be explicit about scope.**
   - They must clearly separate:
     - in scope
     - out of scope
     - open questions
     - assumptions
     - risks

8. **Design docs must be operational, not aspirational.**
   - They must be written so that downstream implementation, review, QA, and documentation work can proceed without guessing.


9. **Design docs must be concise.**
   - Keep the document as short as possible while preserving clarity and completeness.
   - Do not include unnecessary background, repeated explanations, or long narrative discussion.
   - Prefer structured sections, bullet points, and explicit decisions over lengthy prose.
   - Include only information needed for implementation, review, QA, CI/CD, dependency handling, installation, and documentation updates.
   - If a topic is still unresolved, state the open question briefly instead of expanding speculative discussion.
---

## Branching and Change Management

1. **Do not work directly on protected branches.**
   - Never directly push to `main`, `master`, or any protected release branch.

2. **Use isolated working branches.**
   - Changes should be made in a dedicated branch or equivalent isolated workspace.
   - Working branch name must include a clear agent signature.

3. **Keep changes focused.**
   - One branch or pull request should correspond to one primary objective.

4. **Do not merge unreviewed work through informal shortcuts.**
   - Changes must follow the repository’s normal review and integration process.

5. **Keep branch state current as required by workflow.**
   - Sync with the current integration base before finalizing work when needed.

---

## GitHub Workflow Rules

### Identity and Signature Rules
1. **All GitHub-facing messages must include a clear agent signature.**
   - This applies to:
     - commit messages
     - pull request titles or descriptions
     - issue titles or descriptions where appropriate
     - comments
     - review comments
     - other GitHub messages or updates
   - The signature must make the message source unambiguous to other agents and reviewers.

2. **Do not post unsigned GitHub messages.**
   - Other agents must be able to identify the message author immediately.

### GitHub Tooling Rules
3. **Always use GitHub CLI (`gh`) to create GitHub issues and pull requests.**
   - Do not create issues or PRs through ad hoc manual workflows if `gh` is available.
   - Prefer consistent, scriptable, auditable GitHub operations.

### Issue / PR Traceability Rules
4. **Every GitHub pull request must have a corresponding GitHub issue.**
   - No PR may exist without an associated issue.
   - If no suitable issue exists, create one first.

5. **Every PR must clearly reference its corresponding issue.**
   - The relationship between issue and PR must be explicit and easy to audit.

6. **If a PR is merged, its corresponding issue must be closed.**
   - Merged work must not leave its tracking issue open.
   - Issue lifecycle and PR lifecycle must remain aligned.

### Review Request Rules
7. **When a PR is ready for review, the appropriate reviewer must be explicitly tagged with `@`.**
   - Do not assume passive visibility is sufficient.
   - Review requests must be explicit.

8. **When follow-up changes are needed after review, explicitly tag the PR author with `@`.**
   - Do not leave ambiguous requests.
   - The person expected to act must be clearly identified.

### Status Transition Rules
9. **A task or issue may be moved to `in_review` only when both of the following are true:**
   - a PR has already been created
   - the appropriate reviewer has already been explicitly tagged on that PR

10. **Do not mark work as `in_review` prematurely.**
    - Review state begins only after an actual review handoff is complete.

---

## Safety and Risk Control

### High-Risk Changes
Use extra caution when touching:
- authentication or authorization logic
- secrets or credential handling
- deployment configuration
- CI/CD workflows
- dependency manifests or lockfiles
- database schema or migrations
- shared interfaces, contracts, or core configuration

For high-risk changes:
- make the minimum necessary modification
- call out the risk explicitly
- avoid combining them with unrelated work

### Destructive Actions
Do not perform destructive actions casually, including:
- deleting files without clear reason
- rewriting history
- force-pushing shared branches
- removing tests, safeguards, or validation without explicit justification

---

## Communication Rules

All agents should communicate in a way that is:
- concise
- explicit
- factual
- operationally useful

When reporting work, include:
- what was changed
- what was not changed
- what was validated
- what remains uncertain or risky

Avoid:
- unnecessary chatter
- vague claims
- overstating confidence
- hiding blockers or unresolved issues

### Reply Discipline
1. **Do not casually reply to teammates’ messages.**
   - Unnecessary back-and-forth creates noise and confusion.

2. **Only reply when at least one of the following is true:**
   - you were explicitly tagged
   - your task or work was explicitly mentioned
   - a direct response from you is operationally required

3. **Do not reply just to acknowledge, agree, or add low-value commentary.**
   - Avoid "noted", "sounds good", "thanks", or similar low-signal responses unless explicitly required.

4. **If clarification is needed, start a new thread.**
   - Do not derail an existing thread unless your participation is explicitly required there.
   - Clarification requests must be targeted and minimal.

5. **Default to silence unless a response is necessary for progress.**
   - Communication must serve execution, coordination, or risk reduction.

---

## Definition of Done

Work is only considered done when:
- the requested scope has been addressed
- unrelated scope has not been silently included
- relevant validation has been performed or explicitly omitted with explanation
- risks, assumptions, and limitations have been disclosed
- the result is in a state suitable for normal review and integration
- any required GitHub issue / PR linkage has been completed correctly
- review handoff has followed the required tagging and status rules
- any required design doc has been completed and received final human review