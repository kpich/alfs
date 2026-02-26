"""Fetch and subsample top-level English Wikibooks titles.

Usage:
    python -m alfadict.etl.fetch_titles --num-docs 10 --seed 42 --output titles.txt
"""

import argparse
import random

import requests

API_URL = "https://en.wikibooks.org/w/api.php"
HEADERS = {"User-Agent": "alfadict/0.1 (https://github.com/alfadict/alfadict; bot)"}


def fetch_all_titles() -> list[str]:
    titles: list[str] = []
    params: dict = {
        "action": "query",
        "list": "allpages",
        "apnamespace": 0,
        "aplimit": 500,
        "format": "json",
    }

    while True:
        response = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()

        for page in data["query"]["allpages"]:
            title = page["title"]
            if "/" not in title:
                titles.append(title)

        cont = data.get("continue")
        if cont is None:
            break
        params.update(cont)

    return titles


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and subsample Wikibooks titles")
    parser.add_argument(
        "--num-docs", type=int, required=True, help="Number of titles to sample"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--output", required=True, help="Output file path (one title per line)"
    )
    args = parser.parse_args()

    print("Fetching all top-level Wikibooks titles...")
    all_titles = fetch_all_titles()
    print(f"Found {len(all_titles)} top-level titles")

    rng = random.Random(args.seed)
    sampled = rng.sample(all_titles, min(args.num_docs, len(all_titles)))
    print(f"Sampled {len(sampled)} titles")

    with open(args.output, "w") as f:
        for title in sampled:
            f.write(title + "\n")

    print(f"Wrote titles to {args.output}")


if __name__ == "__main__":
    main()
