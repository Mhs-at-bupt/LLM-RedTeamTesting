# Defense Paper Adaptation Notes

This project now provides **paper-aligned default configurations** for each defense module.
Implementation remains engineering-oriented, but key algorithmic steps and default knobs follow the original papers.

## 1) Keyword / Rule Filter

- Role: low-cost first-pass safety gate.
- Nature: engineering baseline, not a standalone paper reimplementation.

## 2) SmoothLLM (paper-aligned defaults)

- Paper idea retained: create randomized perturbation copies of a prompt and aggregate robustness via majority vote.
- This repo implementation:
  - character-level perturbation family (`swap`, `patch`, `insert`),
  - multiple copies (`num_copies`) with random perturbation ratio,
  - refusal-vs-jailbreak majority vote with configurable threshold.
  - config path: `configs/defense_config.yaml -> smoothllm`.

## 3) SemanticSmooth (paper-aligned defaults)

- Paper idea retained: semantic-equivalent transformed copies and aggregation for robust decision.
- This repo implementation:
  - seven transformations from the paper:
    `spellcheck`, `verb_tense`, `synonym`, `translate`, `summarize`, `paraphrase`, `format`,
  - configurable target translation language (default `French`),
  - multi-variant voting on refusal/jailbreak labels, with judge scores as auxiliary trace.
  - config path: `configs/defense_config.yaml -> semantic_smooth`.

## 4) JailGuard-style Detector (paper-aligned defaults)

- Paper idea retained: attacks are less robust than benign prompts under input mutation.
- This repo implementation:
  - 18 text mutators (paper-consistent mutator count),
  - mutator combination policy (`combination_size`) for stronger generalization,
  - discrepancy aggregation from length dispersion, uniqueness, and lexical distance.
  - config path: `configs/defense_config.yaml -> jailguard`.

## 5) Integrated Pipeline

- Final decision chain: `Keyword -> SmoothLLM -> SemanticSmooth -> JailGuard`.
- Returns per-layer trace metadata for reproducibility and ablation analysis.

## 6) AutoDAN Refusal Prefix Expansion

- Refusal keyword/prefix list is expanded to AutoDAN-style ASR checking length.
- Applied to:
  - `src/llm_redteam/evaluation/refusal_detector.py`
  - `configs/defense_config.yaml -> keyword_filter.refusal_patterns`
- The expanded list includes common refusal starts and policy-declination phrases such as:
  - `I'm sorry`, `I apologize`, `As an AI`, `I cannot`, `I can't assist`,
    `is illegal and unethical`, `not appropriate`, `not legal`, etc.

## Conclusion

Defense modules are integrated with paper-aligned defaults and a unified interface for reproducible graduation-project experiments.
For strict paper reproduction, use the original benchmark datasets and full experimental protocol from each paper.
