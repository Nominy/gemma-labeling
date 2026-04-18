from __future__ import annotations

import argparse

import uvicorn

from gemma_labeling.config import DEFAULT_HOST, DEFAULT_PORT, TAXONOMY_PATH
from gemma_labeling.model import GemmaModelRuntime
from gemma_labeling.service import LabelingService
from gemma_labeling.taxonomy import Taxonomy
from gemma_labeling.web import create_app


def build_app():
    taxonomy = Taxonomy.from_yaml(TAXONOMY_PATH)
    runtime = GemmaModelRuntime(taxonomy)
    service = LabelingService(taxonomy, runtime)
    return create_app(service)


def cli() -> None:
    parser = argparse.ArgumentParser(description="Run the Gemma labeling PoC.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "gemma_labeling.main:build_app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    cli()
