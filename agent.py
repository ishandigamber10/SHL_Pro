"""
agent.py
=========================================================
Conversational Recommendation Agent for SHL assessments.

Responsibilities
----------------
- Interpret user hiring needs from a stateless message history
- Ask clarifying questions when the request is too vague
- Retrieve relevant SHL assessments through the retriever layer
- Detect mid-conversation refinements and update results
- Compare assessments grounded in catalog data
- Refuse out-of-scope and prompt-injection requests
- Return structured outputs for the FastAPI layer
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from groq import Groq

import retriever

# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables.")

client = Groq(api_key=GROQ_API_KEY)

# =========================================================
# LOGGING
# =========================================================

log = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================

MIN_QUERY_LENGTH = 12
DEFAULT_RESULTS = 10
MAX_RECOMMENDATIONS = 10
LLM_CONTEXT_HITS = 5

REFINEMENT_TERMS = [
    "actually",
    "instead",
    "also",
    "add",
    "remove",
    "exclude",
    "change",
    "update",
    "more",
    "less",
]

COMPARE_PATTERNS = [
    "difference between",
    "compare",
    "vs",
    "versus",
]

OUT_OF_SCOPE_PATTERNS = [
    "legal advice",
    "salary negotiation",
    "labor law",
    "politics",
    "ignore previous instructions",
    "reveal your prompt",
    "system prompt",
    "developer message",
    "jailbreak",
    "override instructions",
    "prompt injection",
]

VAGUE_PATTERNS = [
    "i need a test",
    "i need an assessment",
    "need assessment",
    "suggest assessment",
    "help me hire",
    "recommend test",
    "need hiring test",
]

# =========================================================
# HELPERS
# =========================================================

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _combined_user_query(messages: list[dict]) -> str:
    """
    Combine all user messages to preserve stateless conversation context.
    """
    parts: list[str] = []
    for msg in messages:
        if msg.get("role") == "user":
            content = _normalize(str(msg.get("content", "")))
            if content:
                parts.append(content)
    return _normalize(" ".join(parts))


def _latest_user_message(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "user":
            return _normalize(str(msg.get("content", "")))
    return ""


def _needs_clarification(query: str) -> bool:
    q = query.lower().strip()

    if len(q) < MIN_QUERY_LENGTH:
        return True

    return any(pattern in q for pattern in VAGUE_PATTERNS)


def _is_refinement(query: str) -> bool:
    q = query.lower()
    return any(term in q for term in REFINEMENT_TERMS)


def _is_compare_query(query: str) -> bool:
    q = query.lower()
    return any(pattern in q for pattern in COMPARE_PATTERNS)


def _is_out_of_scope(query: str) -> bool:
    q = query.lower()
    return any(pattern in q for pattern in OUT_OF_SCOPE_PATTERNS)


def _extract_comparison_targets(query: str) -> tuple[Optional[str], Optional[str]]:
    """
    Try to extract two assessment names from compare-style queries.
    Examples:
      - "difference between OPQ and GSA"
      - "Compare OPQ vs GSA"
      - "OPQ versus GSA"
    """
    q = _normalize(query)

    patterns = [
        r"difference between\s+(.+?)\s+and\s+(.+?)(?:\?|$)",
        r"compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\?|$)",
        r"(.+?)\s+vs\.?\s+(.+?)(?:\?|$)",
        r"(.+?)\s+versus\s+(.+?)(?:\?|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, q, flags=re.IGNORECASE)
        if match:
            left = _normalize(match.group(1)).strip(" \"'`")
            right = _normalize(match.group(2)).strip(" \"'`")
            if left and right:
                return left, right

    return None, None


def _safe_test_type(hit: dict) -> str:
    test_types = hit.get("test_types") or []
    if isinstance(test_types, list) and test_types:
        return str(test_types[0])
    return "General"


def _dedupe_hits(hits: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []

    for hit in hits:
        url = str(hit.get("url", "")).strip()
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(hit)

    return unique


def _catalog_lookup(name: str) -> Optional[dict]:
    """
    Try exact/fuzzy lookup first, then fall back to semantic retrieval.
    """
    if not name:
        return None

    hit = retriever.lookup_by_name(name)
    if hit:
        return hit

    fallback_hits = retriever.search(name, n_results=1)
    return fallback_hits[0] if fallback_hits else None


def _build_llm_context(hits: list[dict]) -> str:
    """
    Compact grounding context for the LLM.
    """
    lines: list[str] = []
    for hit in hits[:LLM_CONTEXT_HITS]:
        name = hit.get("name", "")
        test_types = ", ".join(hit.get("test_types", []) or [])
        job_levels = ", ".join(hit.get("job_levels", []) or [])
        duration = hit.get("duration", "")
        description = _normalize(str(hit.get("description", "")))
        if description:
            description = description[:350]

        lines.append(
            "\n".join(
                [
                    f"Name: {name}",
                    f"Test types: {test_types or 'General'}",
                    f"Job levels: {job_levels or 'N/A'}",
                    f"Duration: {duration or 'N/A'}",
                    f"Description: {description or 'N/A'}",
                ]
            )
        )

    return "\n\n".join(lines)


# =========================================================
# RESPONSE GENERATION
# =========================================================

def generate_llm_response(query: str, hits: list[dict], refinement: bool = False) -> str:
    if not hits:
        return (
            "I could not find suitable SHL assessments for that requirement. "
            "Please share the job title, seniority, and the key skills you want to assess."
        )

    context = _build_llm_context(hits)

    prefix = ""
    if refinement:
        prefix = "Updated recommendations based on your revised constraints.\n\n"

    prompt = f"""
You are an SHL assessment recommendation assistant.

STRICT RULES:
- Use ONLY the assessments provided below.
- Never invent assessments.
- Keep the response concise and practical.
- Maximum 120 words.
- Mention why the assessments fit the role.
- Prefer a balanced mix of technical and behavioral coverage when relevant.
- Do NOT explain generic assessment theory.
- Do NOT mention assessments not in the provided list.

User hiring requirement:
{query}

Retrieved assessments:
{context}

Write:
1. A short recommendation summary.
2. A brief explanation of fit.
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_tokens=180,
    )

    content = ""
    if completion and completion.choices:
        content = completion.choices[0].message.content or ""

    content = _normalize(content)
    if not content:
        content = (
            "I found relevant SHL assessments, but I could not generate a response summary."
        )

    return prefix + content


def compare_assessments(query: str) -> str:
    """
    Compare assessments grounded in catalog data.
    """
    left_name, right_name = _extract_comparison_targets(query)

    left_hit = _catalog_lookup(left_name) if left_name else None
    right_hit = _catalog_lookup(right_name) if right_name else None

    # If we couldn't extract both names, fall back to the top two retrieval hits.
    if not left_hit or not right_hit:
        fallback_hits = retriever.search(query, n_results=4)
        fallback_hits = _dedupe_hits(fallback_hits)

        if left_hit and right_hit:
            hits = [left_hit, right_hit]
        elif len(fallback_hits) >= 2:
            hits = fallback_hits[:2]
        else:
            return (
                "I could not identify both assessments clearly from the catalog. "
                "Please name the two assessments you want to compare."
            )
    else:
        hits = [left_hit, right_hit]

    context_blocks: list[str] = []
    for hit in hits:
        name = hit.get("name", "")
        test_types = ", ".join(hit.get("test_types", []) or [])
        job_levels = ", ".join(hit.get("job_levels", []) or [])
        duration = hit.get("duration", "")
        description = _normalize(str(hit.get("description", "")))
        if description:
            description = description[:450]

        context_blocks.append(
            "\n".join(
                [
                    f"Name: {name}",
                    f"Test types: {test_types or 'General'}",
                    f"Job levels: {job_levels or 'N/A'}",
                    f"Duration: {duration or 'N/A'}",
                    f"Description: {description or 'N/A'}",
                ]
            )
        )

    prompt = f"""
You are comparing two SHL assessments.

STRICT RULES:
- Use ONLY the catalog evidence provided below.
- Do not invent details.
- Keep the response concise and grounded.
- Explain the primary purpose, differences, and best use cases.
- If one assessment has limited evidence, say so clearly.

Question:
{query}

Catalog evidence:
{chr(10).join(context_blocks)}

Write a short, grounded comparison.
"""

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.2,
        max_tokens=220,
    )

    content = ""
    if completion and completion.choices:
        content = completion.choices[0].message.content or ""

    content = _normalize(content)
    if not content:
        content = "I could not generate a comparison from the catalog evidence."

    return content


# =========================================================
# MAIN AGENT
# =========================================================

def respond(messages: list[dict]) -> dict[str, Any]:
    """
    Stateless agent entry point.
    """
    query = _combined_user_query(messages)
    latest_user = _latest_user_message(messages)

    log.info("Agent received query: %r", query)

    if not query:
        return {
            "reply": (
                "Could you share the hiring role or the kind of assessment you need?"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    if _is_out_of_scope(query):
        return {
            "reply": (
                "I can only help with SHL assessment recommendations, assessment comparisons, "
                "and catalog-related queries."
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    if _is_compare_query(query):
        reply = compare_assessments(query)
        return {
            "reply": reply,
            "recommendations": [],
            "end_of_conversation": False,
        }

    if _needs_clarification(latest_user or query):
        return {
            "reply": (
                "Could you share a bit more detail about the role?\n\n"
                "Helpful details include:\n"
                "- job title\n"
                "- seniority level\n"
                "- technical skills\n"
                "- behavioral skills\n"
                "- remote testing needs"
            ),
            "recommendations": [],
            "end_of_conversation": False,
        }

    is_refinement = _is_refinement(query)

    try:
        hits = retriever.search(
            query=query,
            n_results=DEFAULT_RESULTS,
        )
    except Exception as e:
        log.exception("Retriever failure: %s", e)
        return {
            "reply": (
                "The retrieval system encountered an internal error while searching for assessments."
            ),
            "recommendations": [],
            "end_of_conversation": True,
        }

    valid_hits = []
    for hit in _dedupe_hits(hits):
        url = str(hit.get("url", "")).strip()
        if retriever.is_valid_catalog_url(url):
            valid_hits.append(hit)

    valid_hits = valid_hits[:MAX_RECOMMENDATIONS]

    reply = generate_llm_response(
        query=query,
        hits=valid_hits,
        refinement=is_refinement,
    )

    recommendations: list[dict[str, Any]] = []
    for hit in valid_hits:
        recommendations.append(
            {
                "name": hit.get("name", ""),
                "url": hit.get("url", ""),
                "test_type": _safe_test_type(hit),
            }
        )

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": False,
    }


if __name__ == "__main__":
    # Simple local smoke test if needed
    sample_messages = [
        {"role": "user", "content": "Need a Java developer assessment with reasoning skills"},
        {"role": "assistant", "content": "Sure. What is the seniority level?"},
        {"role": "user", "content": "Mid-level, around 4 years"},
    ]
    print(respond(sample_messages))