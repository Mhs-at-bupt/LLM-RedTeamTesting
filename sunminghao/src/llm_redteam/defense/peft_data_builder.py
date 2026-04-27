from __future__ import annotations

from pathlib import Path

from llm_redteam.utils.io_utils import load_jsonl, save_jsonl


class PEFTDataBuilder:
    def build(self, attack_results_path: str, output_path: str) -> None:
        rows = load_jsonl(attack_results_path)
        data = []
        for r in rows:
            goal = r.get("goal", "")
            resp = r.get("response", "")
            success = bool(r.get("success", False))
            if success:
                data.append({"instruction": goal, "response": "I cannot assist with harmful or unsafe requests.", "label": "adversarial"})
                data.append({"instruction": goal, "response": "I can help with legal, safe alternatives instead.", "label": "pseudo_compliance"})
            else:
                data.append({"instruction": goal, "response": resp or "I can help with general safe information.", "label": "benign_hard_negative"})
        data.append({"instruction": "Explain gradient descent simply.", "response": "Gradient descent iteratively reduces loss by updating parameters.", "label": "benign"})
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        save_jsonl(data, output_path)

