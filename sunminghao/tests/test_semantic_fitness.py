from llm_redteam.evaluation.semantic_fitness import SemanticFitnessScorer


class MockEncoder:
    def encode(self, text: str) -> list[float]:
        n = max(1, len(text))
        return [float(n % 7), float((n + 3) % 11), float((n + 5) % 13)]


def test_semantic_score_float_and_range() -> None:
    scorer = SemanticFitnessScorer(encoder=MockEncoder())
    s = scorer.score("safe learning", "safe learning response")
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0

