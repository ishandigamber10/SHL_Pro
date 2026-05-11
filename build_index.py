"""
build_index.py
=========================================================
Professional SHL Semantic Retrieval Index Builder
=========================================================

Purpose
-------
Builds a semantic vector index from catalog.json using:
- sentence-transformers
- ChromaDB

This index powers:
- semantic search
- recommendation retrieval
- conversational matching
- Recall@10 evaluation

Outputs
-------
chroma_db/

Features
--------
✅ Persistent ChromaDB storage
✅ Metadata-aware retrieval
✅ Clean embedding pipeline
✅ Duplicate-safe indexing
✅ Resume-safe rebuild
✅ Professional logging
✅ Recruiter-friendly structure

Install
-------
pip install chromadb sentence-transformers torch

Run
---
python build_index.py
"""

from __future__ import annotations

import json
import logging
import shutil
import retriever
from pathlib import Path
from chromadb.config import Settings
import chromadb
from chromadb.config import Settings
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TaskProgressColumn,
)
from sentence_transformers import SentenceTransformer

# =========================================================
# CONFIG
# =========================================================

CATALOG_PATH = Path("catalog.json")

CHROMA_DIR = Path("chroma_db")

COLLECTION_NAME = "shl_assessments"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

if __name__ == "__main__":

    log.info(
        "Building SHL vector index..."
    )

    retriever.build_index()

    log.info(
        "Vector index build complete."
    )

console = Console()


# =========================================================
# HELPERS
# =========================================================

def clean_text(text: str) -> str:
    """
    Normalize whitespace.
    """

    return " ".join(str(text).split()).strip()


def load_catalog() -> list[dict]:
    """
    Load catalog.json safely.
    """

    if not CATALOG_PATH.exists():

        raise FileNotFoundError(
            f"{CATALOG_PATH} does not exist."
        )

    with open(
        CATALOG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    if not isinstance(data, list):

        raise ValueError(
            "catalog.json must contain a list."
        )

    return data


def build_document(entry: dict) -> str:
    """
    Create embedding text for retrieval.
    """

    name = clean_text(
        entry.get("name", "")
    )

    description = clean_text(
        entry.get("description", "")
    )

    job_levels = ", ".join(
        entry.get("job_levels", [])
    )

    test_types = ", ".join(
        entry.get("test_types", [])
    )

    duration = clean_text(
        entry.get("duration", "")
    )

    languages = ", ".join(
        entry.get("languages", [])
    )

    document = f"""
Assessment Name:
{name}

Description:
{description}

Job Levels:
{job_levels}

Test Types:
{test_types}

Duration:
{duration}

Languages:
{languages}
"""

    return clean_text(document)


def validate_entry(entry: dict) -> bool:
    """
    Ensure entry contains minimum viable data.
    """

    if not entry.get("name"):
        return False

    if not entry.get("url"):
        return False

    if not entry.get("description"):
        return False

    return True

# =========================================================
# MAIN
# =========================================================

def build_index():

    # -----------------------------------------------------
    # Load catalog
    # -----------------------------------------------------

    console.print(
        "\n[bold cyan]Loading catalog...[/bold cyan]"
    )

    catalog = load_catalog()

    console.print(
        f"Loaded {len(catalog)} assessments."
    )

    # -----------------------------------------------------
    # Remove existing DB
    # -----------------------------------------------------

    print("Preparing Chroma collection...")

    try:
        client.delete_collection("shl_catalog")
        print("Deleted previous collection.")
    except Exception:
        print("No previous collection found.") 

    # -----------------------------------------------------
    # Initialize embedding model
    # -----------------------------------------------------

    console.print(
        "\n[bold cyan]Loading embedding model...[/bold cyan]"
    )

    model = SentenceTransformer(
        EMBED_MODEL
    )

    console.print(
        f"Using model: [green]{EMBED_MODEL}[/green]"
    )

    # -----------------------------------------------------
    # Initialize ChromaDB
    # -----------------------------------------------------

    client = chromadb.PersistentClient(
        path="chroma_db",
        settings=Settings(
        anonymized_telemetry=False
        )
    )
    try:
        client.delete_collection("shl_assessments")
        print("Deleted old collection.")
    except Exception:
        print("No previous collection found.")

    collection = client.get_or_create_collection(
        name="shl_assessments"
    )


    # -----------------------------------------------------
    # Prepare data
    # -----------------------------------------------------

    ids = []
    docs = []
    metadatas = []

    skipped = 0

    console.print(
        "\n[bold cyan]Preparing documents...[/bold cyan]"
    )

    for idx, entry in enumerate(catalog):

        if not validate_entry(entry):

            skipped += 1

            continue

        doc_id = (
            entry["url"]
            .rstrip("/")
            .split("/")[-1]
        )

        document = build_document(entry)

        metadata = {
            "name": entry.get("name", ""),
            "url": entry.get("url", ""),
            "duration": entry.get("duration", ""),
            "test_types": ", ".join(
                entry.get("test_types", [])
            ),
            "job_levels": ", ".join(
                entry.get("job_levels", [])
            ),
        }

        ids.append(doc_id)
        docs.append(document)
        metadatas.append(metadata)

    console.print(
        f"Prepared {len(docs)} documents."
    )

    if skipped:

        console.print(
            f"[yellow]Skipped {skipped} "
            f"invalid entries.[/yellow]"
        )

    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    console.print(
        "\n[bold cyan]Generating embeddings...[/bold cyan]"
    )

    embeddings = model.encode(
        docs,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # -----------------------------------------------------
    # Store in ChromaDB
    # -----------------------------------------------------

    console.print(
        "\n[bold cyan]Writing vectors to ChromaDB..."
        "[/bold cyan]"
    )

    BATCH_SIZE = 32

    with Progress(
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}"
        ),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:

        task = progress.add_task(
            "Indexing",
            total=len(ids)
        )

        for start in range(
            0,
            len(ids),
            BATCH_SIZE
        ):

            end = start + BATCH_SIZE

            collection.add(
                ids=ids[start:end],
                documents=docs[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end].tolist(),
            )

            progress.update(
                task,
                advance=min(
                    BATCH_SIZE,
                    len(ids) - start
                )
            )

    # -----------------------------------------------------
    # Final stats
    # -----------------------------------------------------

    count = collection.count()

    console.print(
        "\n[bold green]"
        f"✓ Vector index built successfully"
        "[/bold green]"
    )

    console.print(
        f"\nCollection Name : {COLLECTION_NAME}"
    )

    console.print(
        f"Documents Indexed : {count}"
    )

    console.print(
        f"Embedding Model : {EMBED_MODEL}"
    )

    console.print(
        f"Database Path : {CHROMA_DIR}"
    )

    # -----------------------------------------------------
    # Example query test
    # -----------------------------------------------------

    console.print(
        "\n[bold cyan]Running retrieval sanity check..."
        "[/bold cyan]"
    )

    sample_query = (
        "Java developer assessment "
        "with communication skills"
    )

    query_embedding = model.encode(
        sample_query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
    )

    console.print(
        "\n[bold]Sample Query:[/bold]"
    )

    console.print(
        f"  {sample_query}"
    )

    console.print(
        "\n[bold]Top Matches:[/bold]"
    )

    docs_meta = results.get("metadatas", [[]])[0]

    for idx, meta in enumerate(
        docs_meta,
        start=1
    ):

        console.print(
            f"\n  {idx}. "
            f"[cyan]{meta.get('name')}[/cyan]"
        )

        console.print(
            f"     URL: {meta.get('url')}"
        )

        console.print(
            f"     Levels: "
            f"{meta.get('job_levels')}"
        )

        console.print(
            f"     Types: "
            f"{meta.get('test_types')}"
        )

# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":

    build_index()