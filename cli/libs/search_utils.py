from pathlib import Path
import json
from typing import TypedDict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/"data"
MOVIES_PATH = DATA_PATH/"movies.json"
STOPWORDS_PATH = DATA_PATH/"stopwords.txt"
CACHE_PATH = PROJECT_ROOT/"cache"
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_SEARCH_LIMIT = 5
SCORE_PRECISION = 3
DEFAULT_CHUNK_SIZE = 200


def load_movies() -> list[Movie]:
    with open(MOVIES_PATH, "r") as fp:
        data = json.load(fp)

    return data['movies']


def load_stop_words() -> list[str]:

    with open(STOPWORDS_PATH, "r") as file:
        words = file.read().splitlines()
    return words


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]


class GoldenTestCase(TypedDict):
    query: str
    relevant_docs: list[str]


class GoldenDataset(TypedDict):
    test_cases: list[GoldenTestCase]


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    """Create standardized search result

    Args:
        doc_id: Document ID
        title: Document title
        document: Display text (usually short description)
        score: Relevance/similarity score
        **metadata: Additional metadata to include

    Returns:
        Dictionary representation of search result
    """
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
