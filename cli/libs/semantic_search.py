from sentence_transformers import SentenceTransformer
from libs.search_utils import Movie
import numpy as np
from numpy.typing import NDArray
from typing import Any
import os

from libs.search_utils import (
    CACHE_PATH,
    load_movies
)

from pathlib import Path

EmbeddingArray = NDArray[Any]


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: EmbeddingArray | None = None
        self.documents: list[Movie] | None = None
        self.document_map: dict[int, Movie] = {}
        self.embedding_path = os.path.join(CACHE_PATH, "movie_embeddings.npy")

    def generate_embedding(self, text: str) -> EmbeddingArray:
        if not text or not text.strip():
            raise ValueError("Text should not be empty")
        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        self.document_map = {}
        movies_strings: list[str] = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movies_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(
            movies_strings, show_progress_bar=True)
        np.save(self.embedding_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        emb_path = Path(self.embedding_path)
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            if self.embeddings is not None and len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)


def verify_model() -> None:
    ss = SemanticSearch()
    print(f"Model loaded: {ss}")
    print(f"Max sequence length: {ss.model.max_seq_length}")


def embed_text(text):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    ss = SemanticSearch()
    documents = load_movies()
    embeddings = ss.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
