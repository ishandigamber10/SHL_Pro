"""
scraper.py
===========
Scrapes every Individual Test Solution from the SHL product catalog using
Playwright (handles JS-rendered tables) + BeautifulSoup (parses extracted HTML).

Output: catalog.json

Usage:
    # First time only — install browser binary:
    playwright install chromium

    # Run scraper:
    python scraper.py

The scraper is resume-safe: if catalog.json already exists it skips
assessments already scraped and picks up from where it left off.
"""

import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL        = "https://www.shl.com"
CATALOG_BASE    = "https://www.shl.com/products/product-catalog/"
OUTPUT_FILE     = Path("catalog.json")
PAGE_DELAY_MS   = 1200          # ms to wait after each listing page navigation
DETAIL_DELAY_S  = 0.8           # seconds between detail page fetches
HEADLESS        = True          # set False to watch the browser during dev
MAX_RETRIES = 3

BAD_DESC_PATTERNS = [
    "privacy-friendly website",
    "cookie",
    "advertising effectiveness",
    "social media",
    "viewing preferences",
    "opt out",
]

console = Console()
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

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

KNOWN_JOB_LEVELS = [
    "Director", "Entry-Level", "Executive", "Front Line Manager",
    "General Population", "Graduate", "Manager", "Mid-Professional",
    "Professional Individual Contributor", "Supervisor",
]


# ── URL builder ───────────────────────────────────────────────────────────────
def listing_url(start: int) -> str:
    return (
        f"{CATALOG_BASE}?action_doFilteringForm=Search"
        f"&f=1&start={start}&type=1"
    )


# ── Listing page parser ───────────────────────────────────────────────────────
def parse_listing(html: str) -> tuple[list[dict], bool]:
    """
    Parse the Individual Test Solutions table from a listing page.
    Returns (list_of_stubs, has_next_page).
    """
    soup = BeautifulSoup(html, "lxml")
    stubs: list[dict] = []

    # Find the Individual Test Solutions table.
    # There are two tables: Pre-packaged Job Solutions and Individual Test Solutions.
    # We identify ours by looking for the table that follows the "Individual Test Solutions"
    # text node anywhere in the page.
    individual_table = None
    for table in soup.find_all("table"):
        # Walk backwards through siblings/parents to find a header
        prev_text = ""
        for sibling in table.find_all_previous(string=True):
            stripped = sibling.strip()
            if stripped:
                prev_text = stripped
                break
        if "Individual Test" in prev_text or "Individual Test" in table.get_text():
            individual_table = table
            # Don't break – take the LAST such table (the actual individual table,
            # not the one labelled in a header above the pre-packaged table)
    
    # Fallback: if still None, take the last table on the page
    if individual_table is None:
        tables = soup.find_all("table")
        individual_table = tables[-1] if tables else None

    if individual_table is None:
        return stubs, False

    rows = individual_table.find_all("tr")
    for row in rows[1:]:    # skip header row
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        a_tag = cells[0].find("a")
        if not a_tag:
            continue

        name = a_tag.get_text(strip=True)
        href = a_tag.get("href", "")
        url  = urljoin(BASE_URL, href) if href else ""
        if not url or not name:
            continue

        # Column 1: Remote Testing (img present = Yes)
        remote = bool(cells[1].find("img") or cells[1].get_text(strip=True))

        # Column 2: Adaptive/IRT
        adaptive = bool(cells[2].find("img") or cells[2].get_text(strip=True))

        # Column 3: Test type codes  e.g. "A", "K P", "B K"
        raw_types  = cells[3].get_text(strip=True)
        test_types = [t for t in raw_types.split() if t in TYPE_LEGEND]

        stubs.append({
            "name":           name,
            "url":            url,
            "remote_testing": remote,
            "adaptive_irt":   adaptive,
            "test_types":     test_types,
        })

    # Check for a Next link
    next_link = soup.find("a", string=re.compile(r"^Next$", re.I))
    has_next  = next_link is not None

    return stubs, has_next


# ── Detail page parser ────────────────────────────────────────────────────────
def parse_detail(html: str) -> dict:
    """
    Extract clean metadata from SHL assessment detail page.
    """

    soup = BeautifulSoup(html, "lxml")

    details = {
        "description": "",
        "job_levels": [],
        "languages": [],
        "duration": "",
    }

    # -----------------------------------------------------
    # Remove obvious junk
    # -----------------------------------------------------

    for bad in soup.select(
        "script, style, noscript, iframe, "
        "header, footer, nav, svg"
    ):
        bad.extract()

    # -----------------------------------------------------
    # Cookie/privacy junk filters
    # -----------------------------------------------------

    BAD_PATTERNS = [
        "privacy-friendly website",
        "opt out of our cookies",
        "viewing preferences",
        "advertising effectiveness",
        "social media sites",
        "collecting site analytics",
        "cookie usage",
        "third parties",
        "personalization",
        "consent",
        "your region",
        "region that you are in",
    ]

    # -----------------------------------------------------
    # Candidate content containers
    # -----------------------------------------------------

    candidate_sections = []

    selectors = [
        "main",
        "article",
        ".product-catalogue__content",
        ".product-catalogue-training-calendar",
        ".region-content",
        ".field--name-body",
        ".content",
    ]

    for selector in selectors:
        candidate_sections.extend(
            soup.select(selector)
        )

    if not candidate_sections:
        candidate_sections = [soup]

    # -----------------------------------------------------
    # Description extraction
    # -----------------------------------------------------

    candidate_paragraphs = []

    for section in candidate_sections:

        for p in section.find_all("p"):

            text = p.get_text(
                " ",
                strip=True
            )

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            if len(text) < 120:
                continue

            lower = text.lower()

            # Skip junk
            if any(
                bad in lower
                for bad in BAD_PATTERNS
            ):
                continue

            # Skip support/contact fluff
            if lower.startswith(
                (
                    "client support",
                    "candidate support",
                    "learn more",
                )
            ):
                continue

            candidate_paragraphs.append(text)

    # Choose best description
    if candidate_paragraphs:

        details["description"] = max(
            candidate_paragraphs,
            key=len
        )

    # -----------------------------------------------------
    # Full text
    # -----------------------------------------------------

    full = soup.get_text(
        " ",
        strip=True
    )

    full = re.sub(
        r"\s+",
        " ",
        full
    ).strip()

    # -----------------------------------------------------
    # Duration extraction
    # -----------------------------------------------------

    duration_match = re.search(
        r"(\d+)\s*(minutes|minute|min)",
        full,
        re.I
    )

    if duration_match:

        details["duration"] = (
            duration_match.group(0)
        )

    # -----------------------------------------------------
    # Job level extraction
    # -----------------------------------------------------

    found_levels = []

    for level in KNOWN_JOB_LEVELS:

        if level.lower() in full.lower():

            found_levels.append(level)

    details["job_levels"] = found_levels

    # -----------------------------------------------------
    # Language extraction
    # -----------------------------------------------------

    language_match = re.search(
        r"Languages?:\s*([A-Za-z,\-\s]+)",
        full,
        re.I
    )

    if language_match:

        raw = language_match.group(1)

        langs = []

        for lang in raw.split(","):

            lang = lang.strip()

            if not lang:
                continue

            lower = lang.lower()

            if any(
                bad in lower
                for bad in BAD_PATTERNS
            ):
                continue

            # Skip absurd extractions
            if len(lang) > 30:
                continue

            langs.append(lang)

        details["languages"] = langs[:20]

    return details

# ── Slug dedup helper ─────────────────────────────────────────────────────────
def url_slug(url: str) -> str:
    return url.rstrip("/").split("/")[-1]


# ── Save helper ───────────────────────────────────────────────────────────────
def save(catalog: list[dict]) -> None:
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)


# ── Main ──────────────────────────────────────────────────────────────────────
def scrape() -> list[dict]:
    # Load existing progress
    existing: dict[str, dict] = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for item in json.load(f):
                existing[url_slug(item["url"])] = item
        console.print(f"[dim]Resuming — {len(existing)} entries already scraped.[/dim]")

    all_stubs: list[dict] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=HEADLESS)
        ctx     = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = ctx.new_page()

        # ── Phase 1: collect all stubs from listing pages ─────────────────────
        console.print("\n[bold cyan]Phase 1: Collecting listing pages …[/bold cyan]")
        start    = 0
        page_num = 1

        while True:
            url = listing_url(start)
            console.print(f"  Page {page_num} (start={start})", end=" … ")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_timeout(1500)
            except PWTimeout:
                console.print("[red]timeout[/red]")
                break

            html              = page.content()
            stubs, has_next   = parse_listing(html)
            console.print(f"[green]{len(stubs)} items[/green]")

            if not stubs:
                console.print("  [yellow]No items found — stopping pagination.[/yellow]")
                break

            all_stubs.extend(stubs)

            if not has_next:
                console.print("  [dim]Last page reached.[/dim]")
                break

            start    += 12
            page_num += 1
            page.wait_for_timeout(PAGE_DELAY_MS)

        # Deduplicate stubs
        seen: set[str]        = set()
        unique: list[dict]    = []
        for s in all_stubs:
            slug = url_slug(s["url"])
            if slug not in seen:
                seen.add(slug)
                unique.append(s)

        console.print(f"\n[bold]Total unique stubs: {len(unique)}[/bold]")

        # ── Phase 2: enrich each stub with detail page ────────────────────────
        console.print("\n[bold cyan]Phase 2: Fetching detail pages …[/bold cyan]")

        catalog      = list(existing.values())
        already_done = set(existing.keys())
        pending      = [s for s in unique if url_slug(s["url"]) not in already_done]

        console.print(f"  {len(pending)} to fetch, {len(already_done)} already done.\n")

        for stub in tqdm(pending, desc="Detail pages", unit="page"):
            slug = url_slug(stub["url"])

            try:
                page.goto(stub["url"], wait_until="domcontentloaded", timeout=20_000)
                page.wait_for_timeout(600)   # brief settle
                html    = page.content()
                details = parse_detail(html)
            except Exception as e:
                log.warning("Failed detail for %r: %s", stub["name"], e)
                details = {"description": "", "job_levels": [], "languages": [], "duration": ""}

            entry = {
                "name":           stub["name"],
                "url":            stub["url"],
                "test_types":     stub["test_types"],
                "remote_testing": stub["remote_testing"],
                "adaptive_irt":   stub["adaptive_irt"],
                "description":    details["description"],
                "job_levels":     details["job_levels"],
                "languages":      details["languages"],
                "duration":       details["duration"],
            }

            catalog.append(entry)
            already_done.add(slug)
            save(catalog)

            time.sleep(DETAIL_DELAY_S)

        browser.close()

    console.print(f"\n[bold green]✓ Scrape complete — {len(catalog)} assessments saved to {OUTPUT_FILE}[/bold green]")
    return catalog


# ── Stats printer ─────────────────────────────────────────────────────────────
def print_stats(catalog: list[dict]) -> None:
    from collections import Counter
    console.print("\n[bold]── Catalog stats ──────────────────────────────[/bold]")
    console.print(f"  Total assessments : {len(catalog)}")

    type_counts: Counter = Counter()
    for e in catalog:
        for t in e.get("test_types", []):
            type_counts[t] += 1

    console.print("\n  [bold]Test type breakdown:[/bold]")
    for code, count in sorted(type_counts.items()):
        bar  = "█" * (count // 5)
        name = TYPE_LEGEND.get(code, "?")
        console.print(f"    {code}  {name:<35} {count:>4}  {bar}")

    no_desc  = sum(1 for e in catalog if not e.get("description"))
    no_level = sum(1 for e in catalog if not e.get("job_levels"))
    console.print(f"\n  Missing description : {no_desc}/{len(catalog)}")
    console.print(f"  Missing job_levels  : {no_level}/{len(catalog)}")
    console.print("\n  [dim]Sample entries:[/dim]")
    for e in catalog[:3]:
        console.print(f"    [cyan]{e['name']}[/cyan]")
        console.print(f"      {e['url']}")
        console.print(f"      Types={e['test_types']}  Levels={e['job_levels'][:2]}")
        console.print(f"      {e['description'][:100]}…")


if __name__ == "__main__":
    catalog = scrape()
    print_stats(catalog)