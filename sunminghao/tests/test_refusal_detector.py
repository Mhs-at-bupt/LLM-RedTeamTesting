from llm_redteam.evaluation.refusal_detector import RefusalDetector


def test_refusal_patterns_detected() -> None:
    det = RefusalDetector()
    assert det.detect("I'm sorry, I cannot help with that request.")
    assert det.detect("As an AI, I can't provide that.")
    assert det.detect("I’m really sorry, but I cannot assist with that.")


def test_helpful_response_not_refusal() -> None:
    det = RefusalDetector()
    assert not det.detect("Newton's second law states F = m * a.")
