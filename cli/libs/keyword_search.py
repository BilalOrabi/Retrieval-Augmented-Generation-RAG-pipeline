from libs.search_utils import load_movies, load_stop_words
import string
from nltk.stem import PorterStemmer  # type: ignore[import-untyped]


def clean_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = "".join(char for char in cleaned if char not in string.punctuation)
    return cleaned


def tokenize_text(text: str) -> list[str]:
    text = clean_text(text)
    stop_words = load_stop_words()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in text.split() if token and token not in stop_words]
    return tokens


def has_matching_token(query_tokens: list[str], movie_token: list[str]) -> bool:
    for que_token in query_tokens:
        for mov_token in movie_token:
            if que_token in mov_token:
                return True
    return False


def search_command(query: str, n_result: int) -> list[dict]:

    result = []
    movies = load_movies()
    cleaned_query = clean_text(query)
    query_tokens = tokenize_text(cleaned_query)

    for movie in movies:
        movie_tokens = tokenize_text(movie['title'])
        if has_matching_token(query_tokens, movie_tokens):
            result.append(movie)

        if len(result) == n_result:
            break

    return result
