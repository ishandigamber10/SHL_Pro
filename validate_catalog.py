"""
validate_catalog.py
=========================================================
Professional SHL Catalog Validator
=========================================================

Purpose
-------
Validates catalog.json before using it in:
- retrieval systems
- embeddings
- FastAPI APIs
- recommendation pipelines

Checks
------
✅ Required fields exist
✅ URL format validity
✅ Duplicate URL detection
✅ Valid SHL test type codes
✅ Empty field analysis
✅ Completeness sanity check
✅ Statistics summary

Usage
-----
python validate_catalog.py
python validate_catalog.py catalog.json
"""

import json
import sys
from pathlib import Path
from collections import Counter

# =========================================================
# CONFIG
# =========================================================

CATALOG_PATH = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else Path("catalog.json")
)

# Adjust based on actual scrape size
MIN_EXPECTED = 40

REQUIRED_KEYS = {
    "name",
    "url",
    "test_types",
    "remote_testing",
    "adaptive_irt",
    "description",
}

VALID_TYPES = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "K",
    "P",
    "S",
}

VALID_URL_PREFIX = (
    "https://www.shl.com/products/product-catalog/view/"
)

LEGEND = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

# =========================================================
# HELPERS
# =========================================================

errors = []
warnings = []

def err(msg):
    errors.append(msg)

def warn(msg):
    warnings.append(msg)

# =========================================================
# LOAD FILE
# =========================================================

if not CATALOG_PATH.exists():

    print(
        f"\nERROR: {CATALOG_PATH} does not exist.\n"
    )

    sys.exit(1)

try:

    with open(
        CATALOG_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        catalog = json.load(f)

except Exception as e:

    print(
        f"\nERROR: Could not read JSON -> {e}\n"
    )

    sys.exit(1)

# =========================================================
# BASIC STRUCTURE
# =========================================================

if not isinstance(catalog, list):

    print(
        "\nERROR: catalog.json must contain a LIST.\n"
    )

    sys.exit(1)

print(
    f"\nValidating {CATALOG_PATH} "
    f"({len(catalog)} entries)...\n"
)

# =========================================================
# FIELD VALIDATION
# =========================================================

for idx, entry in enumerate(catalog):

    if not isinstance(entry, dict):

        err(
            f"Entry {idx} is not a dictionary"
        )

        continue

    missing = REQUIRED_KEYS - set(entry.keys())

    if missing:

        err(
            f"Entry {idx} "
            f"({entry.get('name', '?')!r}) "
            f"missing fields: {sorted(missing)}"
        )

# =========================================================
# URL VALIDATION
# =========================================================

seen_urls = set()

for idx, entry in enumerate(catalog):

    url = entry.get("url", "").strip()

    if not url:

        err(
            f"Entry {idx} missing URL"
        )

        continue

    if not url.startswith(VALID_URL_PREFIX):

        err(
            f"Invalid SHL URL "
            f"({entry.get('name','?')}): {url}"
        )

    if url in seen_urls:

        err(
            f"Duplicate URL: {url}"
        )

    seen_urls.add(url)

# =========================================================
# TEST TYPE VALIDATION
# =========================================================

for idx, entry in enumerate(catalog):

    test_types = entry.get("test_types", [])

    if not isinstance(test_types, list):

        err(
            f"{entry.get('name','?')} "
            f"has invalid test_types format"
        )

        continue

    for t in test_types:

        if t not in VALID_TYPES:

            err(
                f"Unknown test_type {t!r} "
                f"in {entry.get('name','?')}"
            )

# =========================================================
# COMPLETENESS CHECK
# =========================================================

if len(catalog) < MIN_EXPECTED:

    warn(
        f"Only {len(catalog)} entries found. "
        f"Expected at least {MIN_EXPECTED}. "
        f"Scrape may be incomplete."
    )

# =========================================================
# DESCRIPTION ANALYSIS
# =========================================================

empty_desc = sum(
    1 for e in catalog
    if not e.get("description")
)

if empty_desc:

    warn(
        f"{empty_desc}/{len(catalog)} "
        f"entries have empty descriptions"
    )

# =========================================================
# JOB LEVEL ANALYSIS
# =========================================================

empty_levels = sum(
    1 for e in catalog
    if not e.get("job_levels")
)

if empty_levels:

    warn(
        f"{empty_levels}/{len(catalog)} "
        f"entries missing job_levels"
    )

# =========================================================
# LANGUAGE ANALYSIS
# =========================================================

empty_languages = sum(
    1 for e in catalog
    if not e.get("languages")
)

if empty_languages:

    warn(
        f"{empty_languages}/{len(catalog)} "
        f"entries missing languages"
    )

# =========================================================
# PRINT ERRORS
# =========================================================

print(
    "================================================="
)
print("ERRORS")
print(
    "================================================="
)

if errors:

    for e in errors[:30]:

        print(f"✗ {e}")

    if len(errors) > 30:

        print(
            f"... and {len(errors)-30} more"
        )

else:

    print("✓ No errors found")

# =========================================================
# PRINT WARNINGS
# =========================================================

print("\n=================================================")
print("WARNINGS")
print("=================================================")

if warnings:

    for w in warnings:

        print(f"⚠ {w}")

else:

    print("✓ No warnings")

# =========================================================
# STATISTICS
# =========================================================

print("\n=================================================")
print("STATISTICS")
print("=================================================")

print(f"Total entries : {len(catalog)}")
print(f"Unique URLs   : {len(seen_urls)}")

# ---------------------------------------------------------
# TEST TYPES
# ---------------------------------------------------------

type_counter = Counter()

for entry in catalog:

    for t in entry.get("test_types", []):

        type_counter[t] += 1

print("\nTest Type Distribution:")

for code, count in sorted(type_counter.items()):

    bar = "█" * max(1, count // 5)

    print(
        f"{code:<2} "
        f"{LEGEND.get(code,'?'):<35} "
        f"{count:>4} "
        f"{bar}"
    )

# ---------------------------------------------------------
# JOB LEVELS
# ---------------------------------------------------------

level_counter = Counter()

for entry in catalog:

    for lvl in entry.get("job_levels", []):

        level_counter[lvl] += 1

if level_counter:

    print("\nTop Job Levels:")

    for lvl, cnt in level_counter.most_common(10):

        print(
            f"{lvl:<40} {cnt:>4}"
        )

# ---------------------------------------------------------
# REMOTE / ADAPTIVE
# ---------------------------------------------------------

remote_count = sum(
    1 for e in catalog
    if e.get("remote_testing")
)

adaptive_count = sum(
    1 for e in catalog
    if e.get("adaptive_irt")
)

print(
    f"\nRemote Testing : "
    f"{remote_count}/{len(catalog)}"
)

print(
    f"Adaptive/IRT   : "
    f"{adaptive_count}/{len(catalog)}"
)

# =========================================================
# FINAL STATUS
# =========================================================

print("\n=================================================")

if errors:

    print(
        f"VALIDATION FAILED "
        f"({len(errors)} errors)"
    )

else:

    print("VALIDATION PASSED ✓")

print("=================================================\n")

sys.exit(1 if errors else 0)