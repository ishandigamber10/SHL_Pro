"""
test_scraper.py
================
Small SHL scraper test run (10 entries)

Purpose:
- Verify the catalog page loads correctly
- Parse the first 10 assessment entries
- Fetch their detail pages
- Print clean sample output
- Save results to catalog_test.json

Run:
    py test_scraper.py
"""

import json
import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
OUTPUT_FILE = Path("catalog_test.json")

MAX_ENTRIES = 10
HEADLESS = False
PAGE_DELAY_MS = 3000
DETAIL_DELAY_SEC = 1

BAD_DESCRIPTION_PATTERNS = [
    "privacy-friendly website",
    "opt out of our cookies",
    "keeping track of your viewing preferences",
    "cookie usage",
    "personalization and measuring advertising effectiveness",
    "collecting site analytics",
    "social media sites",
    "third parties",
    "cookie policy",
]

KNOWN_JOB_LEVELS = [
    "Entry-Level",
    "Graduate",
    "Manager",
    "Director",
    "Executive",
    "Supervisor",
    "Mid-Professional",
    "Professional Individual Contributor",
    "Front Line Manager",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# =========================================================
# FETCH
# =========================================================

def fetch(page, url):
    """Load a page and return rendered HTML as BeautifulSoup."""
    try:
        page.goto(url, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_timeout(PAGE_DELAY_MS)
        html = page.content()
        return BeautifulSoup(html, "lxml")
    except Exception as e:
        log.error("Failed to fetch %s -> %s", url, e)
        return None


# =========================================================
# LISTING PARSER
# =========================================================

def parse_listing_page(soup):
    """
    Extract catalog entry links from the listing page.
    Returns a list of dicts with name and url.
    """
    items = []
    seen = set()

    if soup is None:
        return items

    for link in soup.find_all("a", href=True):
        href = link.get("href", "")
        text = link.get_text(" ", strip=True)

        if not text:
            continue

        if "/products/product-catalog/view/" not in href:
            continue

        full_url = urljoin(BASE_URL, href)

        if full_url in seen:
            continue

        seen.add(full_url)

        items.append({
            "name": text,
            "url": full_url,
        })

    return items


# =========================================================
# DETAIL PARSER
# =========================================================

def clean_text(text):
    return re.sub(r"\s+", " ", text).replace("\xa0", " ").strip()


def parse_detail_page(soup):
    """
    Extract a clean description and a few useful fields from the detail page.
    """
    details = {
        "description": "",
        "job_levels": [],
        "languages": [],
        "duration": "",
    }

    if soup is None:
        return details

    # Remove obvious junk
    for bad in soup.select("script, style, noscript, iframe, header, footer, svg"):
        bad.extract()

    # Save debug HTML if needed
    with open("debug_detail.html", "w", encoding="utf-8") as f:
        f.write(str(soup))

    # Prefer content from main/article containers
    candidate_sections = []
    for selector in [
        "main",
        "article",
        ".product-catalogue__content",
        ".product-catalogue-training-calendar",
        ".region-content",
        ".content",
        ".field--name-body",
    ]:
        candidate_sections.extend(soup.select(selector))

    if not candidate_sections:
        candidate_sections = [soup.body] if soup.body else [soup]

    candidate_paragraphs = []

    for section in candidate_sections:
        for p in section.find_all("p"):
            text = clean_text(p.get_text(" ", strip=True))

            if len(text) < 80:
                continue

            lower = text.lower()
            if any(bad in lower for bad in BAD_DESCRIPTION_PATTERNS):
                continue

            candidate_paragraphs.append(text)

    if candidate_paragraphs:
        # choose the longest meaningful paragraph
        details["description"] = max(candidate_paragraphs, key=len)

    full_text = clean_text(soup.get_text(" ", strip=True))

    # Duration
    m = re.search(r"(\d+)\s*(minutes|minute|min)", full_text, re.I)
    if m:
        details["duration"] = m.group(0)

    # Job levels
    found_levels = []
    lower_full = full_text.lower()
    for level in KNOWN_JOB_LEVELS:
        if level.lower() in lower_full:
            found_levels.append(level)
    details["job_levels"] = found_levels

    # Languages
    m = re.search(r"Languages?:\s*([A-Za-z,\s\-]+)", full_text, re.I)
    if m:
        langs = [x.strip() for x in m.group(1).split(",") if x.strip()]
        details["languages"] = langs[:20]

    return details


# =========================================================
# MAIN TEST
# =========================================================

def run_test():
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=HEADLESS)
    page = browser.new_page()

    results = []

    try:
        log.info("Opening SHL catalog...")
        soup = fetch(page, CATALOG_URL)

        if soup is None:
            log.error("Failed to load catalog page.")
            return

        items = parse_listing_page(soup)
        log.info("Found %d listing items", len(items))

        if not items:
            log.error("No catalog items found. Check selector logic or page rendering.")
            return

        items = items[:MAX_ENTRIES]

        for idx, item in enumerate(items, start=1):
            log.info("[%d/%d] Fetching: %s", idx, len(items), item["name"])

            detail_soup = fetch(page, item["url"])
            if detail_soup is None:
                log.warning("Skipping %s because detail page could not be fetched.", item["name"])
                continue

            details = parse_detail_page(detail_soup)

            entry = {
                "name": item["name"],
                "url": item["url"],
                "description": details["description"],
                "job_levels": details["job_levels"],
                "languages": details["languages"],
                "duration": details["duration"],
            }

            results.append(entry)

            print("\n===================================")
            print(f"NAME      : {entry['name']}")
            print(f"URL       : {entry['url']}")
            print(f"DURATION  : {entry['duration']}")
            print(f"LEVELS    : {entry['job_levels']}")
            print(f"LANGUAGES : {entry['languages']}")
            print(f"DESC      : {entry['description'][:300]}")
            print("===================================\n")

            time.sleep(DETAIL_DELAY_SEC)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        log.info("Saved %d test entries to %s", len(results), OUTPUT_FILE)

    finally:
        browser.close()
        playwright.stop()


if __name__ == "__main__":
    run_test()
    