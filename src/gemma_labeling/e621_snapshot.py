from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

from gemma_labeling.config import DATA_ROOT


E621_BASE_URL = "https://e621.net"
DEFAULT_USER_AGENT = "gemma-labeling-poc/0.1 (by openai-codex)"
MAX_PAGE_SIZE = 320
CHUNK_SIZE = 40

CATEGORY_NAMES = {
    0: "general",
    1: "artist",
    2: "contributor",
    3: "copyright",
    4: "character",
    5: "species",
    6: "invalid",
    7: "meta",
    8: "lore",
}

DEFAULT_CATEGORY_LIMITS = {
    0: 256,
    5: 128,
    7: 48,
    4: 32,
    3: 24,
    8: 16,
}


def fetch_json(path: str, params: dict[str, object], *, user_agent: str) -> list[dict[str, object]]:
    query = urlencode(params, doseq=True)
    request = Request(
        f"{E621_BASE_URL}{path}?{query}",
        headers={"User-Agent": user_agent, "Accept": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def fetch_top_tags(
    *,
    category: int,
    limit: int,
    user_agent: str,
) -> list[dict[str, object]]:
    tags: list[dict[str, object]] = []
    page = 1

    while len(tags) < limit:
        page_size = min(MAX_PAGE_SIZE, limit - len(tags))
        batch = fetch_json(
            "/tags.json",
            {
                "search[category]": category,
                "search[order]": "count",
                "limit": page_size,
                "page": page,
            },
            user_agent=user_agent,
        )
        if not batch:
            break
        tags.extend(batch)
        if len(batch) < page_size:
            break
        page += 1

    return tags[:limit]


def fetch_aliases_for_tags(tag_names: list[str], *, user_agent: str) -> dict[str, list[str]]:
    aliases_by_canonical: dict[str, list[str]] = defaultdict(list)

    for name_chunk in chunked(tag_names, CHUNK_SIZE):
        page = 1
        while True:
            batch = fetch_json(
                "/tag_aliases.json",
                {
                    "search[consequent_name]": ",".join(name_chunk),
                    "search[status]": "active",
                    "limit": MAX_PAGE_SIZE,
                    "page": page,
                },
                user_agent=user_agent,
            )
            if not batch:
                break

            for alias in batch:
                canonical = str(alias["consequent_name"])
                antecedent = str(alias["antecedent_name"])
                aliases_by_canonical[canonical].append(antecedent)

            if len(batch) < MAX_PAGE_SIZE:
                break
            page += 1

    return {
        canonical: sorted(set(aliases))
        for canonical, aliases in aliases_by_canonical.items()
    }


def fetch_implications_for_tags(tag_names: list[str], *, user_agent: str) -> dict[str, list[str]]:
    implications_by_antecedent: dict[str, list[str]] = defaultdict(list)

    for name_chunk in chunked(tag_names, CHUNK_SIZE):
        page = 1
        while True:
            batch = fetch_json(
                "/tag_implications.json",
                {
                    "search[antecedent_name]": ",".join(name_chunk),
                    "search[status]": "active",
                    "limit": MAX_PAGE_SIZE,
                    "page": page,
                },
                user_agent=user_agent,
            )
            if not batch:
                break

            for implication in batch:
                antecedent = str(implication["antecedent_name"])
                consequent = str(implication["consequent_name"])
                implications_by_antecedent[antecedent].append(consequent)

            if len(batch) < MAX_PAGE_SIZE:
                break
            page += 1

    return {
        antecedent: sorted(set(consequents))
        for antecedent, consequents in implications_by_antecedent.items()
    }


def build_snapshot(
    *,
    category_limits: dict[int, int],
    user_agent: str,
) -> dict[str, object]:
    selected_tags: list[dict[str, object]] = []
    seen_names: set[str] = set()

    for category, limit in category_limits.items():
        if limit <= 0:
            continue
        for tag in fetch_top_tags(category=category, limit=limit, user_agent=user_agent):
            name = str(tag["name"])
            if name in seen_names:
                continue
            selected_tags.append(tag)
            seen_names.add(name)

    canonical_names = [str(tag["name"]) for tag in selected_tags]
    alias_map = fetch_aliases_for_tags(canonical_names, user_agent=user_agent)
    implication_map = fetch_implications_for_tags(canonical_names, user_agent=user_agent)

    rules: list[dict[str, object]] = []
    canonical_set = set(canonical_names)
    for tag in selected_tags:
        canonical = str(tag["name"])
        rules.append(
            {
                "canonical": canonical,
                "category": CATEGORY_NAMES.get(int(tag["category"]), f"category_{tag['category']}"),
                "aliases": alias_map.get(canonical, []),
                "implications": [
                    implied
                    for implied in implication_map.get(canonical, [])
                    if implied in canonical_set
                ],
            }
        )

    return {
        "metadata": {
            "source": "e621",
            "fetched_at": datetime.now(UTC).isoformat(),
            "user_agent": user_agent,
            "category_limits": {
                CATEGORY_NAMES.get(category, str(category)): limit
                for category, limit in category_limits.items()
                if limit > 0
            },
            "tag_count": len(rules),
            "base_url": E621_BASE_URL,
        },
        "tags": rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a local e621 taxonomy snapshot.")
    parser.add_argument(
        "--output",
        default=str(DATA_ROOT / "taxonomy" / "e621_tags.yaml"),
        help="Output YAML path.",
    )
    parser.add_argument("--general", type=int, default=DEFAULT_CATEGORY_LIMITS[0])
    parser.add_argument("--species", type=int, default=DEFAULT_CATEGORY_LIMITS[5])
    parser.add_argument("--meta", type=int, default=DEFAULT_CATEGORY_LIMITS[7])
    parser.add_argument("--character", type=int, default=DEFAULT_CATEGORY_LIMITS[4])
    parser.add_argument("--copyright", type=int, default=DEFAULT_CATEGORY_LIMITS[3])
    parser.add_argument("--lore", type=int, default=DEFAULT_CATEGORY_LIMITS[8])
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    args = parser.parse_args()

    category_limits = {
        0: args.general,
        5: args.species,
        7: args.meta,
        4: args.character,
        3: args.copyright,
        8: args.lore,
    }
    snapshot = build_snapshot(category_limits=category_limits, user_agent=args.user_agent)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"Wrote {snapshot['metadata']['tag_count']} tags to {output_path}")


if __name__ == "__main__":
    main()
