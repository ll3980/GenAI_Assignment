"""Verify API behavior against the actual pretrained spaCy model."""

import math

from fastapi.testclient import TestClient
import numpy as np
import pytest
import spacy

from app.main import app, bigram_model
from app.bigram_model import BigramModel

client = TestClient(app)


def test_root_and_openapi():
    assert client.get("/").json() == {"Hello": "World"}
    paths = client.get("/openapi.json").json()["paths"]
    assert "post" in paths["/generate"]
    assert {"get", "post"} <= set(paths["/embedding"])


def test_generate_uses_observed_bigrams():
    response = client.post("/generate", json={"start_word": "THE", "length": 20})
    assert response.status_code == 200
    words = response.json()["generated_text"].split()
    assert words[0] == "the"
    assert 1 <= len(words) <= 20
    for left, right in zip(words, words[1:]):
        assert right in bigram_model.bigram_probs[left]


def test_bigram_probabilities_and_terminal_word():
    # A word may occur both before another word and at the end of a text.
    model = BigramModel(["a b a", "b c"])
    for row in model.bigram_probs.values():
        assert math.isclose(sum(row.values()), 1.0)
    assert model.bigram_probs["a"] == {"b": 1.0}
    assert model.generate_text("c", 20) == "c"


@pytest.mark.parametrize("length", [0, -1, 201, 1.5, "10"])
def test_generate_rejects_invalid_lengths(length):
    assert client.post("/generate", json={"start_word": "the", "length": length}).status_code == 422


def test_generate_rejects_unknown_start():
    assert client.post("/generate", json={"start_word": "pineapple", "length": 10}).status_code == 404


def test_embedding_equals_the_classroom_spacy_operation():
    response = client.post("/embedding", json={"word": "apple"})
    assert response.status_code == 200
    result = response.json()
    assert result["word"] == "apple"
    assert result["model"] == "en_core_web_lg"
    assert result["dimensions"] == 300
    assert result["has_vector"] is True
    assert len(result["embedding"]) == 300
    assert all(math.isfinite(x) for x in result["embedding"])
    assert any(x != 0 for x in result["embedding"])

    # Load the full classroom model independently of the API's cached model.
    expected = spacy.load("en_core_web_lg")("apple").vector
    np.testing.assert_array_equal(result["embedding"], expected)


def test_get_and_post_return_the_same_full_vector():
    get_result = client.get("/embedding", params={"word": "apple"})
    post_result = client.post("/embedding", json={"word": " apple "})
    assert get_result.status_code == post_result.status_code == 200
    assert get_result.json() == post_result.json()


@pytest.mark.parametrize("word", ["", "   ", "apple banana", "123", "apple!", "a" * 101])
def test_embedding_rejects_invalid_word(word):
    assert client.post("/embedding", json={"word": word}).status_code == 422
    assert client.get("/embedding", params={"word": word}).status_code == 422


def test_missing_word_is_rejected():
    assert client.post("/embedding", json={}).status_code == 422
    assert client.get("/embedding").status_code == 422


def test_out_of_vocabulary_word_is_reported():
    word = "zzzzzzqqqqxxxyyyzzzzzz"
    assert client.post("/embedding", json={"word": word}).status_code == 404
    assert client.get("/embedding", params={"word": word}).status_code == 404
