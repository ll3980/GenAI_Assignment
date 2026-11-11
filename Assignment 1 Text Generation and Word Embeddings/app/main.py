"""Module 3 text generation API extended with Module 2 word embeddings."""

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.bigram_model import BigramModel
from app.embedding_model import calculate_embedding

app = FastAPI(
    title="Text Generation and Word Embedding API",
    description="Assignment 1: a bigram generator and spaCy word embeddings.",
    version="1.0.0",
)

# The sample corpus supplied in the Module 3 setup activity.
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str = Field(min_length=1, max_length=100, examples=["the"])
    length: int = Field(default=20, ge=1, le=200, strict=True, examples=[20])


class TextGenerationResponse(BaseModel):
    generated_text: str


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"word": "apple"}]})
    word: str = Field(min_length=1, max_length=100)


class EmbeddingResponse(BaseModel):
    word: str
    model: str
    model_version: str
    dimensions: int
    has_vector: bool
    embedding: list[float]


@app.get("/", tags=["Status"])
def read_root():
    return {"Hello": "World"}


@app.post("/generate", response_model=TextGenerationResponse, tags=["Text generation"])
def generate_text(request: TextGenerationRequest):
    """Preserve the text generation endpoint from the Module 3 activity."""
    try:
        text = bigram_model.generate_text(request.start_word, request.length)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail="The starting word is not in the sample corpus. Try 'the', 'we', or 'bigram'.",
        ) from error
    return {"generated_text": text}


def embedding_response(word: str) -> dict:
    """Share the same embedding calculation between GET and POST."""
    try:
        return calculate_embedding(word)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except KeyError as error:
        raise HTTPException(
            status_code=404,
            detail="No pretrained vector is available for this word. Try a common English word.",
        ) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/embedding", response_model=EmbeddingResponse, tags=["Word embeddings"])
def get_word_embedding(request: EmbeddingRequest):
    """Return all vector dimensions for a word supplied in a JSON body."""
    return embedding_response(request.word)


@app.get("/embedding", response_model=EmbeddingResponse, tags=["Word embeddings"])
def get_word_embedding_query(
    word: Annotated[str, Query(min_length=1, max_length=100, description="One English word")],
):
    """Also allow a direct browser query, for example /embedding?word=apple."""
    return embedding_response(word)
