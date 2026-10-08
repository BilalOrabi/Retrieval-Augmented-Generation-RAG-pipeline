from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT/"data"/"movies.json"
STOP_WORDS_PATH = PROJECT_ROOT/"data"/"stopwords.txt"


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as fp:
        data = json.load(fp)

    return data['movies']


def load_stop_words() -> list[str]:

    with open(STOP_WORDS_PATH, "r") as file:
        words = file.read().splitlines()
    return words
