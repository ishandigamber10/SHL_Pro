# SHL Assessment Recommender

A take-home assignment for the SHL Labs AI Intern role.

The task: build a conversational agent that takes a recruiter from "I need an assessment" to a specific shortlist of SHL Individual Test Solutions, through dialogue. No keyword search. No faceted filtering. Just a conversation where the agent asks what it needs to know and then recommends.

---

## Table of Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Tech stack](#tech-stack)
- [Setup](#setup)
- [Running locally](#running-locally)
- [API reference](#api-reference)
- [Agent behavior](#agent-behavior)
- [Scoring criteria](#scoring-criteria)
- [Deployment on Render](#deployment-on-render)
- [Environment variables](#environment-variables)
- [Why I made these choices](#why-i-made-these-choices)

---

## How it works

The main thing this project had to solve: SHL's catalog has 400+ assessments. A recruiter typing "I need something for a Java developer" has no idea whether they want a knowledge test, a personality questionnaire, or both. The agent figures that out through conversation.

Each call to `POST /chat` runs two LLM calls in sequence:

```
POST /chat  →  full conversation history
                      │
               ROUTER call (temp=0)
               Reads conversation, picks an intent:
               CLARIFY / RECOMMEND / REFINE / COMPARE / REFUSE
                      │
         ┌────────────┼────────────┬────────────┐
      CLARIFY      REFUSE      COMPARE     REC / REFINE
      return       return      name        ChromaDB
      question     refusal     lookup      search
      directly     directly       │            │
                             RESPONDER call (temp=0.3)
                             Sees: conversation + top-10 retrieved entries
                             Returns: JSON with reply + recommendations
                                          │
                                   URL validator
                                   Strips anything not in the retrieved set
                                          │
                                      Response
```

The URL validation step matters most. The responder only sees the catalog entries that ChromaDB retrieved for that specific query. If it tries to return a URL it invented, the validator catches and strips it before anything goes out. Hallucinated URLs don't survive that check.

---

## Project structure

```
shl-recommender/
│
├── scraper.py        # Playwright scraper → catalog.json
├── retriever.py      # ChromaDB index builder + search
├── agent.py          # Router + responder pipeline
├── main.py           # FastAPI app (/health, /chat)
├── build_index.py    # One-time: catalog.json → chroma_db/
├── test_agent.py     # 15 behavior probe tests
│
├── catalog.json      # Scraped assessments (commit this)
├── chroma_db/        # Vector index (built from catalog.json)
│
├── requirements.txt
├── render.yaml
├── .env.example
└── README.md
```

---

## Tech stack

| Layer | Library | Reason |
|---|---|---|
| API | `fastapi` + `uvicorn` | Required by the assignment spec |
| LLM | `groq` — llama-3.3-70b-versatile | Fast enough for the 30s timeout. Two calls per request take ~4s total |
| Embeddings | `sentence-transformers` via `torch` — all-MiniLM-L6-v2 | Runs locally, no API cost, good recall for this catalog |
| Vector store | `chromadb` (persistent, cosine) | No external service needed |
| Scraper | `playwright` + `beautifulsoup4` + `lxml` | SHL's tables are JS-rendered — `requests` alone returns empty HTML |
| Validation | `pydantic` v2 | Schema enforcement on every request and response |
| Dev utilities | `tqdm`, `rich`, `python-dotenv` | Progress bars for scraping, `.env` loading |

---

## Setup

### Prerequisites

- Python 3.10+
- A Groq API key from [console.groq.com](https://console.groq.com) (free)

### Install

```bash
git clone <your-repo-url>
cd shl-recommender

pip install -r requirements.txt
playwright install chromium        # downloads the browser binary — one time only
```

### Configure

```bash
cp .env.example .env
# open .env and set GROQ_API_KEY=your_key_here
```

---

## Running locally

Run these steps in order. Each depends on the previous.

**Step 1 — Scrape the catalog**

```bash
python scraper.py
```

Takes 30–60 minutes. Playwright opens a real browser, paginates through the catalog (12 items per page), and visits each assessment's detail page for description, job levels, languages, and duration. It saves after every single detail page — if it crashes at item 200, re-running picks up from there.

Expected output: `catalog.json` with 400+ entries:

```json
{
  "name": "Verify - Numerical Reasoning (New)",
  "url": "https://www.shl.com/products/product-catalog/view/verify-numerical-reasoning-new/",
  "test_types": ["A"],
  "remote_testing": true,
  "adaptive_irt": false,
  "description": "Measures the ability to make correct decisions...",
  "job_levels": ["Graduate", "Mid-Professional", "Manager"],
  "languages": ["English", "French", "German"],
  "duration": "17 minutes"
}
```

**Step 2 — Build the vector index**

```bash
python build_index.py
```

Embeds every catalog entry with `all-MiniLM-L6-v2` and writes the ChromaDB index to `chroma_db/`. Takes 1–2 minutes. Only needs re-running if `catalog.json` changes.

**Step 3 — Run the behavior tests**

```bash
python test_agent.py
```

Runs 15 probes against the live agent — schema compliance, clarification behavior, recommendation quality, refinement handling, comparison grounding, off-topic refusal, and URL integrity. Fix anything failing before deploying.

**Step 4 — Start the API**

```bash
uvicorn main:app --reload --port 8000
```

Swagger UI at `http://localhost:8000/docs`.

**Step 5 — Health check**

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## API reference

### `GET /health`

Returns `200` once the index is loaded. Per the assignment spec, the evaluator allows up to 2 minutes on cold start.

```json
{"status": "ok"}
```

### `POST /chat`

Stateless. Every call must include the full conversation history.

**Request**

```json
{
  "messages": [
    {"role": "user",      "content": "I am hiring a Java developer who works with stakeholders"},
    {"role": "assistant", "content": "What seniority level are you targeting?"},
    {"role": "user",      "content": "Mid-level, around 4 years"}
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `messages` | array | Full history. At least one `user` message required. |
| `messages[].role` | `"user"` or `"assistant"` | |
| `messages[].content` | string | Non-empty. |

**Response**

```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer with stakeholder responsibilities.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "https://www.shl.com/products/product-catalog/view/java-8-new/",
      "test_type": "K"
    },
    {
      "name": "OPQ32r",
      "url": "https://www.shl.com/products/product-catalog/view/opq32r/",
      "test_type": "P"
    }
  ],
  "end_of_conversation": false
}
```

| Field | Type | Notes |
|---|---|---|
| `reply` | string | The agent's response. |
| `recommendations` | array | Empty `[]` when clarifying or refusing. 1–10 items when recommending. |
| `recommendations[].name` | string | Exact name from the catalog. |
| `recommendations[].url` | string | Exact URL from the catalog. Never invented. |
| `recommendations[].test_type` | string | Primary type code — see table below. |
| `end_of_conversation` | boolean | `true` only after the user signals they're done. |

**Test type codes**

| Code | Type |
|---|---|
| A | Ability and Aptitude |
| B | Biodata and Situational Judgement |
| C | Competencies |
| D | Development and 360 |
| E | Assessment Exercises |
| K | Knowledge and Skills |
| P | Personality and Behavior |
| S | Simulations |

**HTTP errors**

| Code | Cause |
|---|---|
| `400` | Bad request schema |
| `500` | Agent error (LLM timeout, index not loaded) |

---

## Agent behavior

### Intent routing

Every request runs two LLM calls. The router (`temp=0`) reads the conversation and decides what to do. It never touches the catalog. The responder gets the retrieved entries and generates the reply.

| Intent | When it fires | What happens |
|---|---|---|
| `CLARIFY` | No role or competency in the query yet | Returns one specific question. `recommendations: []`. |
| `RECOMMEND` | Enough context for a first shortlist | Semantic search → 1–10 results returned. |
| `REFINE` | User adds or changes a constraint | Re-runs search with the updated query. List updates; conversation doesn't reset. |
| `COMPARE` | User names two assessments and asks to compare | Both looked up by name. Comparison generated from catalog data only. |
| `REFUSE` | Off-topic, legal question, or injection attempt | Polite decline. `recommendations: []`. |

### Turn cap

The evaluator stops conversations at 8 turns. The agent commits to a recommendation by turn 6. If it somehow hasn't by turn 8, a force-recommend instruction gets injected and it returns a best-effort shortlist from whatever context exists.

---

## Scoring criteria

The evaluator grades three things:

**Hard evals — must pass, or the submission fails outright**
- Schema valid on every response
- Every URL in `recommendations` exists in the scraped catalog
- Conversation stays within 8 turns

**Recall@10**
- Average fraction of the "correct" assessments that appear in the top-10 shortlist, across all test traces (public + holdout)

**Behavior probes**
- Binary pass/fail per scenario: Does it refuse off-topic questions? Does it clarify before recommending on a vague first message? Does it honor a mid-conversation refinement without resetting?

Run `python test_agent.py` locally to check all 15 behavior probes before submitting.

---

## Deployment on Render

`render.yaml` is in the repo — Render picks it up automatically.

**Steps:**

1. Push to GitHub. Include `catalog.json` and `chroma_db/` in the repo.
2. Render → New Web Service → connect the repo
3. Render → Environment → add `GROQ_API_KEY`
4. Deploy

```yaml
# render.yaml summary
buildCommand: pip install -r requirements.txt && python build_index.py
startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
healthCheckPath: /health
```

One practical issue with Render's free tier: services sleep after inactivity. The evaluator allows 2 minutes for `/health` on cold start, but if you want the service warm between test runs, [UptimeRobot](https://uptimerobot.com) (free) can ping `/health` every 5 minutes.

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Your Groq key |
| `CATALOG_PATH` | No | `catalog.json` | Where the scraped catalog lives |
| `CHROMA_DIR` | No | `chroma_db` | Where the vector index lives |

```bash
cp .env.example .env
# fill in GROQ_API_KEY
```

---

## Why I made these choices

**Groq over OpenAI**

The 30-second timeout drove everything here. A GPT-4o call under load can take 15–20 seconds on its own. Two of those would time out regularly. Groq's llama-3.3-70b-versatile typically returns in 1–3 seconds. Two calls clock around 4–6 seconds total. That's the whole reason — not quality, speed.

**Two LLM calls instead of one**

Routing and responding need different temperatures and you can't set both at once. The router has to be deterministic (`temp=0`) — a probabilistic CLARIFY/RECOMMEND decision means the agent occasionally asks questions when it should recommend, which tanks Recall@10. The responder can be slightly creative at `temp=0.3` for natural replies. Splitting them was the only clean way to get deterministic routing without making every reply sound robotic.

**Catalog context injected as the final message, not in the system prompt**

The system prompt is fixed at startup. The retrieved hits change every query. Injecting the top-10 results as the last message in the conversation puts them right before the next generated token, where the model pays the most attention. Putting all 400 assessments in the system prompt would be slow and noisy — the model would struggle to separate relevant from irrelevant entries.

**Playwright for scraping**

SHL's catalog tables are rendered by JavaScript. Running `requests.get()` on the catalog page returns the page shell — the table rows simply aren't there. You need a real browser. Playwright launches Chromium, waits for `networkidle`, and returns the actual DOM. There's no lighter-weight workaround.

**all-MiniLM-L6-v2 for embeddings**

384 dimensions, runs on CPU, no API cost. The SHL catalog is specific enough that a general-purpose embedding model separates assessments well — "numerical reasoning" and "personality questionnaire" end up far apart in embedding space regardless of model size. A larger model would slow down index builds without improving retrieval in any meaningful way for this use case.

**Cosine similarity in ChromaDB**

Assessment descriptions vary in length — some are a single sentence, some are three paragraphs. Dot product would rank longer ones higher just for having more text. Cosine normalizes for length, so ranking reflects what an assessment measures, not how much its description says.
