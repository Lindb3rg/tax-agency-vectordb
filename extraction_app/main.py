from sentence_transformers import SentenceTransformer
from typing import Union
import numpy as np

# Modellen laddas en gång när modulen importeras
# Undviker att ladda om den vid varje anrop
MODEL_NAME = "intfloat/multilingual-e5-large"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """
    Lazy-loader för embeddingsmodellen.
    Laddar modellen första gången den behövs och cachar den sedan.
    """
    global _model
    if _model is None:
        print(f"Laddar embeddingsmodell: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        print("Modell laddad.")
    return _model


def embed_document(text: str) -> list[float]:
    """
    Används av API 1 (extraktion) när chunks ska lagras i Qdrant.
    multilingual-e5-large förväntar sig prefix "passage: " för dokument.

    Args:
        text: Chunktexten som ska vektoriseras

    Returns:
        En lista med 1024 floats (vektorn)
    """
    model = get_model()
    prefixed = f"passage: {text}"
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()


def embed_query(text: str) -> list[float]:
    """
    Används av API 3 (frågeservice) när en användares fråga ska vektoriseras.
    multilingual-e5-large förväntar sig prefix "query: " för sökfrågor.

    Args:
        text: Användarens fråga

    Returns:
        En lista med 1024 floats (vektorn)
    """
    model = get_model()
    prefixed = f"query: {text}"
    embedding = model.encode(prefixed, normalize_embeddings=True)
    return embedding.tolist()


def embed_batch(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """
    Vektoriserar flera texter på en gång — mer effektivt än ett anrop i taget
    när man processar en hel PDF.

    Args:
        texts:    Lista med texter att vektorisera
        is_query: True om det är sökfrågor, False om det är dokumentchunks

    Returns:
        Lista med vektorer, en per text
    """
    model = get_model()
    prefix = "query: " if is_query else "passage: "
    prefixed_texts = [f"{prefix}{t}" for t in texts]
    embeddings = model.encode(prefixed_texts, normalize_embeddings=True, batch_size=32)
    return embeddings.tolist()