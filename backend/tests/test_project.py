"""Small offline tests for utility functions; they do not call Ollama."""
from app import score_passage, split_text


def test_split_text_returns_overlapping_passages():
    text = "abcdefghijklmnopqrstuvwxyz" * 100
    passages = split_text(text, size=100, overlap=20)
    assert len(passages) > 1
    assert all(passages)
    assert passages[0][80:100] == passages[1][:20]


def test_score_passage_matches_query_terms():
    assert score_passage("neural network", "A neural network can learn patterns") > 0
    assert score_passage("apple", "The document discusses pears and bananas") == 0


def test_stop_words_alone_do_not_match():
    assert score_passage("what is it", "What is it that this document covers") == 0
