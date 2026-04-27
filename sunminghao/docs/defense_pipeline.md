# Defense Pipeline

## Keyword Filtering

- Rule and keyword based risk scoring.
- Blocks high-risk prompts by threshold.

## SmoothLLM

- Generates multiple character-perturbed prompt copies.
- Uses majority response agreement as robustness signal.
- Supports configurable disagreement threshold (`block_threshold`).

## SemanticSmooth

- Generates semantic-equivalent safe rewrites.
- Aggregates responses with judge-assisted scoring.
- Supports configurable templates and block threshold.

## JailGuard-style Detector

- Applies multiple input mutations.
- Uses response discrepancy to flag likely jailbreaks.
- Discrepancy score combines length dispersion and response uniqueness.

## PEFT Safety Hardening

- Builds labeled PEFT data from attack outputs.
- Provides LoRA training skeleton for offline open-source model hardening.

## Integrated Defense

1. Keyword/Rule filtering.
2. SmoothLLM consistency defense.
3. SemanticSmooth consistency defense.
4. Optional JailGuard-style detector.
5. Optional PEFT hardening integration (offline for open-source models).

The integrated pipeline also returns a per-layer decision trace for analysis.
