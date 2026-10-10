import argparse
from libs.keyword_search import (
    search_command,
    build_command
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands")

    search_parser = subparsers.add_parser(
        "search", help="Search movies using keywords")
    subparsers.add_parser("build", help="Build the inverted index")

    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "search":
            results = search_command(args.query, 5)
            for i, result in enumerate(results):
                print(f"{i + 1}. {result['title']}")

        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
