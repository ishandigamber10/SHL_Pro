"""
main.py
=========================================================
Professional FastAPI Backend
SHL Assessment Recommendation System
=========================================================

Features
--------
✅ FastAPI lifespan startup
✅ ChromaDB retrieval warmup
✅ Typed request/response schemas
✅ Strict payload validation
✅ Recommendation grounding validation
✅ Structured logging
✅ Request timing middleware
✅ Global exception handling
✅ Production-safe limits
✅ Stateless conversation architecture
✅ Render/Railway deployment ready

Endpoints
---------
GET  /health
POST /chat

Run
---
uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
)

from fastapi.templating import (
    Jinja2Templates,
)

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

import agent
import retriever

# =========================================================
# ENV
# =========================================================

load_dotenv()



# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s "
        "[%(levelname)s] "
        "%(name)s - %(message)s"
    ),
    datefmt="%H:%M:%S",
)

log = logging.getLogger(__name__)

# =========================================================
# CONSTANTS
# =========================================================

MAX_MESSAGE_CHARS = 4000

MAX_TURNS = 40

SERVICE_NAME = (
    "shl-assessment-recommender"
)

SERVICE_VERSION = "1.0.0"

# =========================================================
# DATA CONTRACT
# =========================================================

@dataclass
class AgentResult:
    """
    Internal typed agent response.
    """

    reply: str

    recommendations: list[dict]

    end_of_conversation: bool

# =========================================================
# LIFESPAN
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    log.info(
        "Loading SHL catalog "
        "and vector index..."
    )

    start = time.time()

    retriever.load_index()

    # -----------------------------------------------------
    # Warmup retrieval
    # -----------------------------------------------------

    try:

        retriever.search(
            "leadership assessment",
            n_results=1,
        )

        log.info(
            "Retrieval warmup successful."
        )

    except Exception as e:

        log.exception(
            "Warmup retrieval failed: %s",
            e,
        )

        raise

    elapsed = (
        time.time() - start
    )

    log.info(
        "Startup complete in %.2fs",
        elapsed,
    )

    yield

    log.info("Shutting down API.")

# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="SHL Assessment Recommender",
    description=(
        "Conversational recommendation API "
        "for SHL Individual Test Solutions."
    ),
    version=SERVICE_VERSION,
    lifespan=lifespan,
)

# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory="templates"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# =========================================================
# REQUEST TIMING MIDDLEWARE
# =========================================================

@app.middleware("http")
async def add_process_time_header(
    request: Request,
    call_next,
):

    start = time.time()

    response = await call_next(
        request
    )

    duration = (
        time.time() - start
    )

    response.headers[
        "X-Process-Time"
    ] = str(
        round(duration, 4)
    )

    return response

# =========================================================
# SCHEMAS
# =========================================================

class Message(BaseModel):

    role: Literal[
        "user",
        "assistant"
    ]

    content: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_CHARS,
    )

    model_config = {
        "extra": "forbid"
    }


class ChatRequest(BaseModel):

    messages: list[Message] = Field(
        ...,
        min_length=1,
        max_length=MAX_TURNS,
    )

    @field_validator("messages")
    @classmethod
    def validate_messages(
        cls,
        messages,
    ):

        if not any(
            m.role == "user"
            for m in messages
        ):

            raise ValueError(
                "At least one user "
                "message is required."
            )

        return messages

    model_config = {
        "extra": "forbid"
    }


class Recommendation(BaseModel):

    name: str

    url: str

    test_type: str

    model_config = {
        "extra": "forbid"
    }


class ChatResponse(BaseModel):

    reply: str

    recommendations: list[
        Recommendation
    ]

    end_of_conversation: bool

    model_config = {
        "extra": "forbid"
    }
# =========================================================
# HOME UI
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="shl_ui.html",
        context={}
    )

# =========================================================
# HEALTH ENDPOINT
# =========================================================

@app.get(
    "/health",
    tags=["ops"],
)
def health():

    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "catalog_size": len(
            retriever._catalog
        ),
        "index_loaded": (
            retriever._collection
            is not None
        ),
    }

# =========================================================
# CHAT ENDPOINT
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["agent"],
)
def chat(
    req: ChatRequest,
    request: Request,
):

    # -----------------------------------------------------
    # Normalize messages
    # -----------------------------------------------------

    messages = [
        {
            "role": m.role,
            "content": m.content,
        }
        for m in req.messages
    ]

    last_user = next(
        (
            m["content"]
            for m in reversed(messages)
            if m["role"] == "user"
        ),
        "",
    )

    log.info(
        "POST /chat | turns=%d | "
        "last_user=%r",
        len(messages),
        last_user[:120],
    )

    # -----------------------------------------------------
    # Agent inference
    # -----------------------------------------------------

    try:

        raw = agent.respond(
            messages
        )

        result = AgentResult(
            reply=str(
                raw.get(
                    "reply",
                    "",
                )
            ),
            recommendations=list(
                raw.get(
                    "recommendations",
                    [],
                )
            ),
            end_of_conversation=bool(
                raw.get(
                    "end_of_conversation",
                    False,
                )
            ),
        )

    except Exception as e:

        log.exception(
            "Agent error: %s",
            e,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "The recommendation "
                "engine encountered "
                "an internal error."
            ),
        )

    # -----------------------------------------------------
    # Recommendation validation
    # -----------------------------------------------------

    recommendations = []

    for rec in result.recommendations:

        try:

            url = str(
                rec.get("url", "")
            )

            # ---------------------------------------------
            # Grounding safeguard
            # ---------------------------------------------

            if not retriever.is_valid_catalog_url(
                url
            ):

                log.warning(
                    "Rejected hallucinated URL: %s",
                    url,
                )

                continue

            recommendations.append(
                Recommendation(
                    name=str(
                        rec.get(
                            "name",
                            "",
                        )
                    ),
                    url=url,
                    test_type=str(
                        rec.get(
                            "test_type",
                            "",
                        )
                    ),
                )
            )

        except Exception:

            log.warning(
                "Dropped malformed "
                "recommendation: %r",
                rec,
            )

    log.info(
        "Response | recs=%d | eoc=%s",
        len(recommendations),
        result.end_of_conversation,
    )

    return ChatResponse(
        reply=result.reply,
        recommendations=recommendations,
        end_of_conversation=(
            result.end_of_conversation
        ),
    )

# =========================================================
# GLOBAL EXCEPTION HANDLER
# =========================================================

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):

    log.exception(
        "Unhandled exception: %s",
        exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": (
                "Internal server error."
            )
        },
    )

# =========================================================
# DEV ENTRYPOINT
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )