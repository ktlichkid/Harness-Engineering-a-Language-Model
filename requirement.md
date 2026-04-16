# Product Requirements Document

## Project Title
Small-Scale LLM Training Framework

---

## 1. Overview

This project aims to build a small-scale but complete language model training framework from scratch, with increasing capability across four milestones.

The goal is not only to train a small model, but to establish a disciplined end-to-end system covering:
- tokenizer and model implementation
- training and checkpointing
- benchmarking and scaling
- data processing and filtering
- evaluation and post-training reasoning improvement

The project is intended as an engineering and research framework, not merely a one-off training script.

---

## 2. Product Goal

The goal of this project is to produce a usable small-scale LLM training stack that can:

- train a language model from scratch
- support reproducible experiments
- scale from single-GPU training to multi-GPU training
- process and improve training data quality
- support reasoning-oriented evaluation and post-training improvement

---

## 3. Development Gating Requirement

The work must be divided into exactly four milestones.

Each milestone must be treated as a hard gate:
- after a milestone is completed, the team must stop
- the result must be handed to a human for review and validation
- no work on the next milestone may begin until explicit human approval is given

This gating requirement is mandatory.

---

## 4. Milestones

## Milestone 1 — Core Training Stack

### Objective
Build the first end-to-end training-capable version of the system.

### Required Deliverables
- BPE tokenizer
- Transformer language model
- cross-entropy loss
- AdamW optimizer
- training loop
- support for serializing and loading model state
- support for serializing and loading optimizer state

### Constraints
- single-GPU support only
- the implementation must not use definitions from:
  - `torch.nn`
  - `torch.nn.functional`
  - `torch.optim`
- exceptions allowed:
  - `torch.nn.Parameter`
  - container classes in `torch.nn` such as `Module`, `ModuleList`, `Sequential`, etc.
  - `torch.optim.Optimizer` base class
- other PyTorch definitions may be used

### Training Data
- TinyStories
- OpenWebText

### Quality Requirement
Milestone 1 must include:
- complete test coverage for the implemented core components
- CI/CD sufficient to validate correctness automatically
- coding-quality safeguards if feasible, including hooks and related quality checks

### Exit Condition
Milestone 1 is complete only when:
- the core training system runs successfully on a single GPU
- checkpoint save/load works
- tests are present and passing
- CI/CD is in place and functioning
- complete documentation
- a human reviews and approves the milestone

---

## Milestone 2 — Performance and Multi-GPU Training

### Objective
Extend the system to support benchmarking, profiling, and distributed training.

### Required Deliverables
- benchmark harness
- profiling harness
- activation checkpointing
- Flash Attention 2 Triton kernel
- distributed data parallel training
- optimizer state sharding
- fully sharded data parallel training

### Capability Requirement
- this milestone must support multi-GPU training

### Exit Condition
Milestone 2 is complete only when:
- performance and profiling infrastructure is usable
- multi-GPU training is operational
- distributed/sharded training functionality is working at the intended level
- complete documentation
- a human reviews and approves the milestone

---

## Milestone 3 — Data Processing Pipeline

### Objective
Build a data pipeline for large-scale raw web text preparation.

### Required Deliverables
- conversion of Common Crawl HTML into text
- filtering of extracted text using multiple methods
- filtering targets should include:
  - harmful content
  - personally identifiable information
  - other low-quality or unsuitable content as appropriate
- deduplication of the training data

### Goal
This milestone should produce cleaner and more usable large-scale pretraining data.

### Exit Condition
Milestone 3 is complete only when:
- HTML-to-text conversion works
- filtering pipeline is implemented and usable
- deduplication is implemented
- the output data is in a form suitable for downstream training
- complete documentation
- a human reviews and approves the milestone

---

## Milestone 4 — Reasoning Evaluation and Post-Training Improvement

### Objective
Add reasoning-focused evaluation and post-training improvement methods.

### Required Deliverables
- zero-shot prompting baseline on the MATH dataset
- supervised fine-tuning using reasoning traces from a stronger reasoning model
- Expert Iteration with verified rewards
- Group-Relative Policy Optimization (GRPO) with verified rewards

### Goal
This milestone should improve reasoning performance beyond base pretraining and establish a framework for reasoning-oriented post-training.

### Exit Condition
Milestone 4 is complete only when:
- the MATH zero-shot baseline is implemented and reported
- supervised fine-tuning is implemented
- Expert Iteration is implemented
- GRPO is implemented
- verified-reward-based reasoning improvement is demonstrated at the intended level
- complete documentation
- a human reviews and approves the milestone

---

## 5. Scope Boundaries

### In Scope
- tokenizer implementation
- language model implementation
- training loop and checkpointing
- benchmarking and profiling
- distributed and sharded training
- web-data extraction and filtering
- deduplication
- reasoning evaluation and post-training improvement

### Out of Scope
- productization for end users
- UI or web application
- inference serving platform
- model deployment platform
- broad model alignment or safety coverage beyond explicitly stated filtering and reward-based methods
- general-purpose MLOps platformization beyond what is needed for the milestones

---

## 6. Success Criteria

This project is successful if:
- each milestone produces a usable, reviewable artifact
- each milestone stops for human validation before continuation
- the system progresses from a working single-GPU trainer to a multi-GPU training stack
- the project includes a usable data pipeline for large-scale training text
- the project includes reasoning-focused evaluation and post-training improvement methods

---

## 7. Key Risks

- milestone boundaries may blur unless kept strict
- performance work may introduce substantial implementation complexity
- data filtering quality may strongly affect downstream model quality
- post-training reasoning improvements may depend heavily on reward design and trace quality
- premature continuation without human checkpoint approval would violate the intended process

---

## 8. Final Requirement

At the end of each milestone:
- stop all forward development
- present the milestone for human validation
- wait for explicit human approval
- only then proceed to the next milestone