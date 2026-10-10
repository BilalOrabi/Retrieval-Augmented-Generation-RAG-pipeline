from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/"data"
MOVIES_PATH = DATA_PATH/"movies.json"
STOPWORDS_PATH = DATA_PATH/"stopwords.txt"
CACHE_PATH = PROJECT_ROOT/"cache"


def load_movies() -> list[dict]:
    with open(MOVIES_PATH, "r") as fp:
        data = json.load(fp)

    return data['movies']


def load_stop_words() -> list[str]:

    with open(STOPWORDS_PATH, "r") as file:
        words = file.read().splitlines()
    return words
