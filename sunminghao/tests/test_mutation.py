from llm_redteam.attacks.mutation import StructuredMutator


def test_character_word_sentence_mutation_runs() -> None:
    m = StructuredMutator()
    text = "Please explain safe machine learning concepts."
    assert isinstance(m.character_mutation(text), str)
    assert isinstance(m.word_mutation(text), str)
    assert isinstance(m.sentence_mutation("First sentence. Second sentence."), str)


def test_benign_not_empty_after_mutation() -> None:
    m = StructuredMutator()
    text = "Hello world"
    out = m.mutate(text)
    assert isinstance(out, str)
    assert out.strip() != ""

