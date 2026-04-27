from __future__ import annotations

import random
import string


class StructuredMutator:
    """Three-level mutation used by enhanced AutoDAN variants."""

    def __init__(self) -> None:
        self.light_obfuscation_map = {"a": "@", "e": "3", "i": "1", "o": "0", "s": "$"}

    def character_mutation(self, text: str) -> str:
        if not text:
            return text
        chars = list(text)
        op = random.choice(["insert", "delete", "replace", "obfuscate"])
        idx = random.randint(0, len(chars) - 1)
        if op == "insert":
            chars.insert(idx, random.choice(string.ascii_lowercase))
        elif op == "delete" and len(chars) > 1:
            chars.pop(idx)
        elif op == "replace":
            chars[idx] = random.choice(string.ascii_lowercase)
        else:
            c = chars[idx].lower()
            chars[idx] = self.light_obfuscation_map.get(c, chars[idx])
        out = "".join(chars).strip()
        return out if out else text

    def word_mutation(self, text: str) -> str:
        words = text.split()
        if not words:
            return text
        idx = random.randint(0, len(words) - 1)
        w = words[idx]
        # Lightweight near-synonym / homophone-like replacements.
        swaps = {
            "use": "utilize",
            "show": "display",
            "build": "construct",
            "to": "too",
            "for": "fore",
            "see": "sea",
        }
        words[idx] = swaps.get(w.lower(), w)
        out = " ".join(words).strip()
        return out if out else text

    def sentence_mutation(self, text: str, use_api_rewriter: bool = False) -> str:
        parts = [x.strip() for x in text.replace("?", ".").replace("!", ".").split(".") if x.strip()]
        if len(parts) <= 1:
            return text
        random.shuffle(parts)
        out = ". ".join(parts) + "."
        return out.strip() if out.strip() else text

    def mutate(
        self,
        text: str,
        p_char: float = 0.2,
        p_word: float = 0.4,
        p_sent: float = 0.4,
    ) -> str:
        r = random.random()
        if r < p_char:
            return self.character_mutation(text)
        if r < p_char + p_word:
            return self.word_mutation(text)
        return self.sentence_mutation(text)

