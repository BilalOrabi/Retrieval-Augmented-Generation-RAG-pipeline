from libs.search_utils import (
    load_movies,
    load_stop_words,
    CACHE_PATH,
    BM25_K1,
    BM25_B,
    DEFAULT_SEARCH_LIMIT,
    Movie,
    SearchResult,
    format_search_result
)
import string
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from collections import defaultdict, Counter
import os
import pickle
import math


class InvertedIndex:
    def __init__(self) -> None:
        self.index: defaultdict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies: defaultdict[int,
                                           Counter[str]] = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")
        self.term_frequencies_path = os.path.join(
            CACHE_PATH, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE_PATH, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0.0
        total_length = 0
        for length in self.doc_lengths.values():
            total_length += length
        return total_length / len(self.doc_lengths)

    def get_documents(self, term: str) -> list[int]:
        return sorted(self.index.get(term, set()))

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie['id']
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self) -> None:

        os.makedirs(CACHE_PATH, exist_ok=True)

        with open(self.index_path, 'wb') as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, 'wb') as file:
            pickle.dump(self.docmap, file)

        with open(self.term_frequencies_path, 'wb') as file:
            pickle.dump(self.term_frequencies, file)

        with open(self.doc_lengths_path, 'wb') as file:
            pickle.dump(self.doc_lengths, file)

    def load(self) -> None:

        with open(self.index_path, 'rb') as file:
            self.index = pickle.load(file)

        with open(self.docmap_path, 'rb') as file:
            self.docmap = pickle.load(file)

        with open(self.term_frequencies_path, 'rb') as file:
            self.term_frequencies = pickle.load(file)

        with open(self.doc_lengths_path, 'rb') as file:
            self.doc_lengths = pickle.load(file)

    def get_tf(self, doc_id: int, term: str) -> int:
        return self.term_frequencies[doc_id][term]

    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        number_of_docs = len(self.docmap)
        df = len(self.index[term])
        return math.log((number_of_docs - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B) -> float:
        tf = self.get_tf(doc_id, term)

        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()

        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1

        return (tf * (k1 + 1)) / (tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        return self.get_bm25_tf(doc_id, term) * self.get_bm25_idf(term)

    def bm25_search(
        self, query: str, limit: int = DEFAULT_SEARCH_LIMIT
    ) -> list[SearchResult]:
        query_tokens = tokenize_text(query)

        scores: dict[int, float] = {}
        for doc_id in self.docmap:
            score = 0.0
            for token in query_tokens:
                score += self.bm25(doc_id, token)
            scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for doc_id, score in sorted_docs[:limit]:
            doc = self.docmap[doc_id]
            formatted_result = format_search_result(
                doc_id=doc["id"],
                title=doc["title"],
                document=doc["description"],
                score=score,
            )
            results.append(formatted_result)

        return results


def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()


def tf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, tokenize_single_term(term))


def idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_idf(tokenize_single_term(term))


def tfidf_command(doc_id: int, term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_tf(doc_id, term) * idx.get_idf(term)


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(tokenize_single_term(term))


def bm25_tf_command(doc_id: int, term: str, k1=BM25_K1, b=BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, tokenize_single_term(term), k1, b)


def bm25search_command(
    query: str, limit: int = DEFAULT_SEARCH_LIMIT
) -> list[SearchResult]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)


def clean_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = "".join(
        char for char in cleaned if char not in string.punctuation)
    return cleaned


stop_words = set(load_stop_words())


def tokenize_text(text: str) -> list[str]:
    stemmer = PorterStemmer()

    text = clean_text(text)
    tokens = [stemmer.stem(token) for token in text.split()
              if token and token not in stop_words]
    return tokens


def tokenize_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:

    inverted_idx = InvertedIndex()
    inverted_idx.load()
    seen, result = set(), []
    query_tokens = tokenize_text(query)

    for query_token in query_tokens:
        matching_doc_ids = inverted_idx.get_documents(query_token)
        for matching_doc_id in matching_doc_ids:

            if matching_doc_id in seen:
                continue

            seen.add(matching_doc_id)
            matching_document = inverted_idx.docmap[matching_doc_id]

            result.append(matching_document)

        if len(result) == limit:
            return result

    return result
