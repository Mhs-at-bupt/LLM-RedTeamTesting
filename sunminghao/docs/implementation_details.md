# Implementation Details

## System Modules

The codebase is organized into:
- `attacks/`: enhanced AutoDAN (joint search, structured mutation, momentum vocabulary, prompt pools).
- `evaluation/`: semantic fitness, refusal detection, judge, metrics.
- `defense/`: keyword filter, SmoothLLM, SemanticSmooth, JailGuard-style detector, PEFT data/training skeleton, integrated pipeline.
- `experiments/`: attack/defense run orchestration and result exporting.

## Enhanced AutoDAN Workflow

1. Initialize user/adversarial prompt pools.
2. Jointly sample candidate user prompts and adversarial prompts.
3. Apply structured mutation (char/word/sentence levels) and momentum-based lexical update.
4. Combine prompts with target goal and query the model.
5. Score via semantic fitness + refusal/judge labels.
6. Update prompt pools and iterate under query budget.

## Joint Search and Structured Mutation

- Joint search keeps user prompt and adversarial prompt optimization in the same iteration.
- Structured mutation supports:
  - Character-level: insert/delete/replace/light obfuscation.
  - Word-level: synonym/near-synonym/homophone-like replacement.
  - Sentence-level: reorder/restructure/paraphrase-style transformation.
- Momentum update follows:
  - `m_t = mu * m_{t-1} + (1 - mu) * score_t`.

