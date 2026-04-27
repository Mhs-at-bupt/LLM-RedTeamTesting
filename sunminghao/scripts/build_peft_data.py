from __future__ import annotations

import argparse

from llm_redteam.defense.peft_data_builder import PEFTDataBuilder


def main(input_path: str = "results/runs/latest/results.jsonl") -> None:
    builder = PEFTDataBuilder()
    builder.build(input_path, "data/private/peft_train.jsonl")
    print("PEFT data generated: data/private/peft_train.jsonl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("build_peft_data")
    parser.add_argument("--input", default="results/runs/latest/results.jsonl")
    args = parser.parse_args()
    main(args.input)
