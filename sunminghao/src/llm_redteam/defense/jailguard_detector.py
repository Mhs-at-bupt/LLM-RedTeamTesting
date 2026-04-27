from __future__ import annotations

import random
import string
from collections.abc import Callable


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


class JailGuardStyleDetector:
    """JailGuard-style discrepancy detector.

    Adaptation note:
    - Keeps the paper intuition (input mutations -> output inconsistency indicates risk).
    - Uses text-length dispersion + lexical dispersion as lightweight discrepancy metrics.
    """

    def __init__(self, config: dict):
        cfg = config.get("jailguard", config)
        self.enabled = bool(cfg.get("enabled", True))
        self.paper_profile = str(cfg.get("paper_profile", "jailguard_v1"))
        self.mutator_count = int(cfg.get("mutator_count", 18))
        self.threshold = float(cfg.get("threshold", 0.25))
        self.char_noise_ratio = float(cfg.get("char_noise_ratio", 0.03))
        self.combination_size = int(cfg.get("combination_size", 6))
        self.seed = cfg.get("seed")
        self.mutators = self._build_mutators()

    def mutate_input(self, prompt: str, mutator_name: str | None = None) -> str:
        if self.seed is not None:
            random.seed(int(self.seed))
        if mutator_name is None:
            mutator_name = random.choice(list(self.mutators))
        fn = self.mutators.get(mutator_name, self._mutator_char_noise)
        return fn(prompt)

    def _build_mutators(self) -> dict[str, Callable[[str], str]]:
        return {
            "word_swap": self._mutator_word_swap,
            "char_noise": self._mutator_char_noise,
            "char_swap": self._mutator_char_swap,
            "char_drop": self._mutator_char_drop,
            "char_insert": self._mutator_char_insert,
            "lowercase": self._mutator_lowercase,
            "uppercase": self._mutator_uppercase,
            "titlecase": self._mutator_titlecase,
            "reverse_words": self._mutator_reverse_words,
            "duplicate_word": self._mutator_duplicate_word,
            "drop_word": self._mutator_drop_word,
            "punctuation_noise": self._mutator_punctuation_noise,
            "whitespace_noise": self._mutator_whitespace_noise,
            "prefix_instruction": self._mutator_prefix_instruction,
            "suffix_instruction": self._mutator_suffix_instruction,
            "bullet_format": self._mutator_bullet_format,
            "question_wrap": self._mutator_question_wrap,
            "translation_hint": self._mutator_translation_hint,
        }

    def _mutator_word_swap(self, prompt: str) -> str:
        words = prompt.split()
        if len(words) < 2:
            return self._mutator_char_noise(prompt)
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
        return " ".join(words)

    def _mutator_char_noise(self, text: str) -> str:
        chars = list(text)
        if not chars:
            return text
        n = max(1, int(len(chars) * self.char_noise_ratio))
        for _ in range(n):
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice(string.ascii_letters + " ")
        return "".join(chars)

    def _mutator_char_swap(self, text: str) -> str:
        chars = list(text)
        if len(chars) < 2:
            return text
        i = random.randint(0, len(chars) - 2)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)

    def _mutator_char_drop(self, text: str) -> str:
        if not text:
            return text
        i = random.randint(0, len(text) - 1)
        return text[:i] + text[i + 1 :]

    def _mutator_char_insert(self, text: str) -> str:
        i = random.randint(0, len(text))
        return text[:i] + random.choice(string.ascii_letters + " ") + text[i:]

    def _mutator_lowercase(self, text: str) -> str:
        return text.lower()

    def _mutator_uppercase(self, text: str) -> str:
        return text.upper()

    def _mutator_titlecase(self, text: str) -> str:
        return text.title()

    def _mutator_reverse_words(self, text: str) -> str:
        words = text.split()
        return " ".join(reversed(words)) if words else text

    def _mutator_duplicate_word(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        i = random.randint(0, len(words) - 1)
        words.insert(i, words[i])
        return " ".join(words)

    def _mutator_drop_word(self, text: str) -> str:
        words = text.split()
        if len(words) < 2:
            return text
        i = random.randint(0, len(words) - 1)
        return " ".join(words[:i] + words[i + 1 :])

    def _mutator_punctuation_noise(self, text: str) -> str:
        punct = random.choice(["!", "?", "...", ",", ";"])
        return f"{text}{punct}"

    def _mutator_whitespace_noise(self, text: str) -> str:
        return "  ".join(text.split())

    def _mutator_prefix_instruction(self, text: str) -> str:
        return f"Please answer carefully and safely: {text}"

    def _mutator_suffix_instruction(self, text: str) -> str:
        return f"{text}\nFollow policy and answer briefly."

    def _mutator_bullet_format(self, text: str) -> str:
        return f"- Request:\n- {text}"

    def _mutator_question_wrap(self, text: str) -> str:
        return f"Could you clarify this request and then respond: {text}?"

    def _mutator_translation_hint(self, text: str) -> str:
        return f"Understand this in another language, then answer in English: {text}"

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(x for x in text.lower().split() if x)

    def compute_response_discrepancy(self, responses: list[str]) -> float:
        if len(responses) < 2:
            return 0.0
        lens = [len(r.strip()) for r in responses]
        mean_len = sum(lens) / len(lens)
        len_dispersion = sum(abs(x - mean_len) for x in lens) / max(1.0, len(lens) * max(1.0, mean_len))
        unique_ratio = len(set(r.strip().lower() for r in responses)) / max(1.0, len(responses))
        pairwise_distances: list[float] = []
        for i in range(len(responses)):
            si = self._token_set(responses[i])
            for j in range(i + 1, len(responses)):
                sj = self._token_set(responses[j])
                union = len(si | sj)
                jaccard = (len(si & sj) / union) if union else 1.0
                pairwise_distances.append(1.0 - jaccard)
        lexical_discrepancy = _safe_mean(pairwise_distances)
        return min(1.0, 0.3 * len_dispersion + 0.3 * unique_ratio + 0.4 * lexical_discrepancy)

    def discrepancy_score(self, prompt: str, model_client) -> float:
        if not self.enabled:
            return 0.0
        mutator_names = list(self.mutators)
        if self.seed is not None:
            random.seed(int(self.seed))
        random.shuffle(mutator_names)
        selected = mutator_names[: max(1, min(self.combination_size, len(mutator_names)))]
        variants: list[str] = []
        while len(variants) < self.mutator_count:
            for name in selected:
                variants.append(self.mutate_input(prompt, name))
                if len(variants) >= self.mutator_count:
                    break
        outputs = [model_client.generate(v).text for v in variants]
        return self.compute_response_discrepancy(outputs)

    def detect(self, prompt: str, model_client) -> bool:
        return self.discrepancy_score(prompt, model_client) >= self.threshold
