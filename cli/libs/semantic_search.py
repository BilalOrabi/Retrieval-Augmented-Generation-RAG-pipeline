from sentence_transformers import SentenceTransformer
from libs.search_utils import Movie
import numpy as np
from numpy.typing import NDArray
from typing import Any, TypedDict
import os

from libs.search_utils import (
    CACHE_PATH,
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    DEFAULT_CHUNK_SIZE
)

from pathlib import Path

EmbeddingArray = NDArray[Any]


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


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

    def search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SemanticSearchResult]:
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        similarities: list[tuple[float, Movie]] = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results: list[SemanticSearchResult] = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results


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


def embed_query_text(query):
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    ss = SemanticSearch()
    documents = load_movies()
    ss.load_or_create_embeddings(documents)

    results = ss.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()


def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i: i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += chunk_size

    return chunks


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    chunks = fixed_size_chunking(text, chunk_size)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
