from sentence_transformers import SentenceTransformer
from libs.search_utils import Movie
import numpy as np
from numpy.typing import NDArray
from typing import Any, TypedDict
import os
import re
import json
from libs.search_utils import (
    CACHE_PATH,
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH
)

from pathlib import Path

EmbeddingArray = NDArray[Any]


class SemanticSearchResult(TypedDict):
    score: float
    title: str
    description: str


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: EmbeddingArray | None = None
        self.chunk_metadata: list[ChunkMetadata] | None = None

    def build_chunk_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(
                text, max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):

                all_chunks.append(chunk)

                chunk_metadata.append(
                    {"movie_idx": idx,
                     "chunk_idx": i,
                     "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(
            all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> EmbeddingArray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)


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


def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                        overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i: i + chunk_size]

        if chunks and len(chunk_words) < + overlap:
            break

        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap

    return chunks


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i: i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")


def embed_chunks_command() -> EmbeddingArray:
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    return searcher.load_or_create_chunk_embeddings(movies)
