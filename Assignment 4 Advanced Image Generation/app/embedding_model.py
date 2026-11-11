"""Return the pretrained spaCy vectors demonstrated in Module 2."""

from functools import lru_cache

import spacy
from spacy.language import Language

MODEL_NAME = "en_core_web_lg"


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Load the English vector model once and reuse it across requests."""
    try:
        # Static word vectors only need the tokenizer and vocabulary.
        # Excluding these components leaves the pretrained vectors unchanged.
        nlp = spacy.load(
            MODEL_NAME,
            exclude=[
                "tok2vec", "tagger", "parser", "attribute_ruler",
                "lemmatizer", "ner",
            ],
        )
    except OSError as error:
        raise RuntimeError(
            "The en_core_web_lg model is unavailable. Run uv sync --locked "
            "from the project directory, then restart the API."
        ) from error
    if nlp.vocab.vectors_length == 0:
        raise RuntimeError("The installed spaCy model has no pretrained word vectors.")
    return nlp


def calculate_embedding(input_word: str) -> dict:
    """Return the full vector for one word as a JSON-compatible dictionary."""
    word = input_word.strip()
    if not word or len(word) > 100 or not word.isalpha():
        raise ValueError("word must be one alphabetic word of at most 100 characters.")

    nlp = get_nlp()
    doc = nlp(word)
    if len(doc) != 1:
        raise ValueError("word must produce exactly one spaCy token.")
    if not doc[0].has_vector or doc[0].vector_norm == 0:
        raise KeyError(word)

    # This is the same nlp(input_word).vector operation as the class demo.
    # tolist() converts NumPy values into JSON-serializable Python numbers.
    vector = doc.vector
    return {
        "word": word,
        "model": MODEL_NAME,
        "model_version": nlp.meta["version"],
        "dimensions": int(vector.shape[0]),
        "has_vector": bool(doc[0].has_vector),
        "embedding": vector.tolist(),
    }
