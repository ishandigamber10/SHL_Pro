"""
retriever.py
=========================================================
Next-Gen SHL Semantic Retrieval Engine
=========================================================
"""

from __future__ import annotations
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import json
import logging
import os
import re
from collections import defaultdict
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
from rich.console import Console
from rich.progress import track

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

log = logging.getLogger(__name__)

console = Console(stderr=True)

# =========================================================
# CONFIG
# =========================================================

CATALOG_PATH = Path(
    os.getenv("CATALOG_PATH", "catalog.json")
)

CHROMA_DIR = Path(
    os.getenv("CHROMA_DIR", "chroma_db")
)

COLLECTION = "shl_assessments_v2"

EMBED_MODEL = "all-MiniLM-L6-v2"

MAX_RESULTS = 10

FETCH_MULTIPLIER = 5

BATCH_SIZE = 64

MAX_DESC_CHARS = 500

MAX_LANGUAGES = 8

MIN_DOC_CHARS = 120

# =========================================================
# QUERY EXPANSION
# =========================================================

QUERY_HINTS = {
    "leadership": (
        "management executive supervisor "
        "decision making people management"
    ),
    "cognitive": (
        "reasoning aptitude analytical "
        "numerical verbal logical"
    ),
    "data analyst": (
        "analytics reasoning problem solving "
        "critical thinking interpretation"
    ),
    "customer service": (
        "communication personality "
        "situational judgement empathy"
    ),
    "developer": (
        "programming software coding "
        "technical knowledge engineering"
    ),
    "java": (
        "backend object oriented programming "
        "software engineering development"
    ),
    "reasoning": (
        "aptitude logical analytical cognitive"
    ),
}

# =========================================================
# TEST TYPE LEGEND
# =========================================================

TYPE_LEGEND = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations",
}

# =========================================================
# GLOBALS
# =========================================================

_client = None

_collection = None

_catalog = []

_url_set = set()

# =========================================================
# CLEAN TEXT
# =========================================================

def clean_text(text: str) -> str:

    if text is None:
        return ""

    text = (
        str(text)
        .encode("utf-8", "ignore")
        .decode("utf-8")
    )

    text = text.replace("â€“", "-")

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()

# =========================================================
# DOCUMENT CREATION
# =========================================================

def _make_document(entry: dict) -> str:

    test_types = ", ".join(
        TYPE_LEGEND.get(t, t)
        for t in entry.get("test_types", [])
    )

    job_levels = ", ".join(
        entry.get("job_levels", [])
    )

    languages = ", ".join(
        entry.get("languages", [])[:MAX_LANGUAGES]
    )

    description = clean_text(
        entry.get("description", "")
    )[:MAX_DESC_CHARS]

    remote = (
        "Supports remote testing."
        if entry.get("remote_testing")
        else ""
    )

    adaptive = (
        "Uses adaptive IRT scoring."
        if entry.get("adaptive_irt")
        else ""
    )

    duration = (
        f"Duration: {entry['duration']}."
        if entry.get("duration")
        else ""
    )

    doc = f"""
    Assessment Name:
    {entry.get('name', '')}

    Test Types:
    {test_types}

    Job Levels:
    {job_levels}

    Languages:
    {languages}

    Description:
    {description}

    {remote}

    {adaptive}

    {duration}
    """

    return clean_text(doc)

# =========================================================
# METADATA
# =========================================================

def _make_metadata(entry: dict) -> dict:

    return {
        "name": entry.get("name", ""),
        "url": entry.get("url", ""),
        "test_types": " ".join(
            entry.get("test_types", [])
        ),
        "job_levels": ", ".join(
            entry.get("job_levels", [])
        ),
        "languages": ", ".join(
            entry.get("languages", [])
        ),
        "duration": entry.get("duration", ""),
        "remote_testing": int(
            bool(entry.get("remote_testing"))
        ),
        "adaptive_irt": int(
            bool(entry.get("adaptive_irt"))
        ),
    }

# =========================================================
# VALIDATION
# =========================================================

def _validate_catalog(data):

    if not isinstance(data, list):
        raise ValueError(
            "catalog.json must contain a list"
        )

    if not data:
        raise ValueError(
            "catalog.json is empty"
        )

# =========================================================
# BUILD INDEX
# =========================================================

def build_index():

    global _client
    global _collection
    global _catalog
    global _url_set

    with open(
        CATALOG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        _catalog = json.load(f)

    _validate_catalog(_catalog)

    _url_set = {
        x["url"]
        for x in _catalog
        if x.get("url")
    }

    embedding_function = (
        SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    )

    _client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:
        _client.delete_collection(COLLECTION)
    except Exception:
        pass

    _collection = _client.create_collection(
        name=COLLECTION,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    entries = []

    for idx, entry in enumerate(_catalog):

        doc = _make_document(entry)

        if len(doc.strip()) < MIN_DOC_CHARS:
            continue

        entries.append(
            (idx, entry, doc)
        )

    for start in track(
        range(0, len(entries), BATCH_SIZE),
        description="Embedding batches",
    ):

        batch = entries[
            start:start + BATCH_SIZE
        ]

        _collection.add(
            ids=[
                str(idx)
                for idx, _, _ in batch
            ],
            documents=[
                doc
                for _, _, doc in batch
            ],
            metadatas=[
                _make_metadata(entry)
                for _, entry, _ in batch
            ],
        )

    console.print(
        "[bold green]✓ Index built successfully[/bold green]"
    )

# =========================================================
# LOAD INDEX
# =========================================================

def load_index():

    global _client
    global _collection
    global _catalog
    global _url_set

    with open(
        CATALOG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        _catalog = json.load(f)

    _validate_catalog(_catalog)

    _url_set = {
        x["url"]
        for x in _catalog
        if x.get("url")
    }

    embedding_function = (
        SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    )

    _client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    try:

        _collection = _client.get_collection(
            COLLECTION,
            embedding_function=embedding_function,
        )

    except Exception:

        log.warning(
            "Collection missing. Rebuilding..."
        )

        build_index()

# =========================================================
# QUERY ENRICHMENT
# =========================================================

def enrich_query(query: str) -> str:

    enriched = f"Job role requirements: {query}"

    lower = query.lower()

    for key, hint in QUERY_HINTS.items():

        if key in lower:
            enriched += f" {hint}"

    return clean_text(enriched)

# =========================================================
# SEARCH
# =========================================================

def search(
    query: str,
    n_results: int = MAX_RESULTS,
):

    if _collection is None:
        raise RuntimeError("Index not loaded.")

    enriched_query = enrich_query(query)

    fetch_n = min(
        n_results * FETCH_MULTIPLIER,
        max(_collection.count(), 1),
    )

    results = _collection.query(
        query_texts=[enriched_query],
        n_results=fetch_n,
        include=["metadatas", "distances"],
    )

    metadatas = results["metadatas"][0]

    distances = results["distances"][0]

    hits = []

    seen_urls = set()

    category_counts = defaultdict(int)

    query_lower = query.lower()

    query_tokens = {
        t
        for t in query_lower.split()
        if len(t) > 3
    }

    for meta, dist in zip(
        metadatas,
        distances
    ):

        url = meta.get("url", "")

        if url in seen_urls:
            continue

        seen_urls.add(url)

        entry_types = (
            meta.get("test_types", "")
            .split()
        )

        title = clean_text(
            meta.get("name", "")
        ).lower()

        title_tokens = set(
            title.split()
        )

        score = max(
            0.0,
            1.0 - float(dist)
        )

        # =================================================
        # Hybrid keyword boost
        # =================================================

        overlap = len(
            query_tokens & title_tokens
        )

        score += overlap * 0.03

        # =================================================
        # Type-aware reranking
        # =================================================

        if (
            "reasoning" in query_lower
            and "A" in entry_types
        ):
            score += 0.12

        if (
            "cognitive" in query_lower
            and "A" in entry_types
        ):
            score += 0.12

        if (
            "personality" in query_lower
            and "P" in entry_types
        ):
            score += 0.12

        if (
            "leadership" in query_lower
            and "C" in entry_types
        ):
            score += 0.08

        if (
            "developer" in query_lower
            and "K" in entry_types
        ):
            score += 0.10

        # =================================================
        # Java boosting
        # =================================================

        if "java" in query_lower:

            if "java" in title:
                score += 0.20

            if "developer" in title:
                score += 0.10

        # =================================================
        # Stakeholder boosting
        # =================================================

        if "stakeholder" in query_lower:

            if "opq" in title:
                score += 0.25

            if "communication" in title:
                score += 0.08

            if "personality" in title:
                score += 0.06
            if "personality" in meta.get(
                "test_types",
    ""
            ).lower():
                score += 0.12    

        # =================================================
        # Generic penalties
        # =================================================

        if "entry level" in title:
            score -= 0.12

        if "industrial" in title:
            score -= 0.08

        # =================================================
        # Diversity penalty
        # =================================================

        primary_type = (
            entry_types[0]
            if entry_types
            else "UNKNOWN"
        )

        category_counts[
            primary_type
        ] += 1

        if (
            category_counts[
                primary_type
            ] > 3
        ):
            score -= 0.04

        hits.append({
            "name": meta.get("name"),
            "url": url,
            "test_types": entry_types,
            "job_levels": meta.get(
                "job_levels",
                ""
            ).split(","),
            "languages": meta.get(
                "languages",
                ""
            ).split(","),
            "duration": meta.get(
                "duration",
                ""
            ),
            "remote_testing": bool(
                meta.get("remote_testing")
            ),
            "adaptive_irt": bool(
                meta.get("adaptive_irt")
            ),
            "_score": round(score, 4),
        })

    hits.sort(
        key=lambda x: x["_score"],
        reverse=True,
    )

    return hits[:n_results]

# =========================================================
# LOOKUP
# =========================================================

def lookup_by_name(name: str):

    lower = name.lower().strip()

    for entry in _catalog:

        if entry["name"].lower() == lower:
            return entry

    hits = search(
        name,
        n_results=1
    )

    return hits[0] if hits else None

# =========================================================
# URL VALIDATION
# =========================================================

def is_valid_catalog_url(
    url: str
) -> bool:

    return url in _url_set

# =========================================================
# SMOKE TEST
# =========================================================

if __name__ == "__main__":

    build_index()

    tests = [
        "Java developer leadership",
        "customer service personality",
        "cognitive reasoning aptitude",
        "data analyst logical reasoning",
    ]

    console.print(
        "\n[bold]Smoke Tests[/bold]"
    )

    for query in tests:

        console.print(
            f"\n[cyan]{query}[/cyan]"
        )

        hits = search(
            query,
            n_results=5,
        )

        for hit in hits:

            console.print(
                f"  [{hit['_score']:.3f}] "
                f"{hit['name']} "
                f"{hit['test_types']}"
            )