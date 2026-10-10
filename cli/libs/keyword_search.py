from libs.search_utils import (
    load_movies,
    load_stop_words,
    CACHE_PATH
)
import string
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]
from collections import defaultdict
import os
import pickle


class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.index_path = os.path.join(CACHE_PATH, "index.pkl")
        self.docmap_path = os.path.join(CACHE_PATH, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        for token in set(tokens):
            self.index[token].add(doc_id)

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

    def load(self) -> None:

        with open(self.index_path, 'rb') as file:
            self.index = pickle.load(file)
        with open(self.docmap_path, 'rb') as file:
            self.docmap = pickle.load(file)


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


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


# def has_matching_token(query_tokens: list[str], movie_token: list[str]) -> bool:
#     for que_token in query_tokens:
#         for mov_token in movie_token:
#             if que_token in mov_token:
#                 return True
#     return False


def search_command(query: str, limit: int = 5) -> list[dict]:

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
