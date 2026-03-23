"""
main.py

FastAPI application exposing the NICE TSD RAG agent as a REST API.

Architecture:
    A single AsyncConnectionPool is opened at startup and shared across:
        - pg_db_utils:        Session CRUD (sessions table)
        - AsyncPostgresSaver: LangGraph conversation checkpointing
          (checkpoints, checkpoint_writes, checkpoint_blobs tables)

    This means a single Postgres database owns all persistent state — sessions,
    conversation history, and LangGraph checkpoints — with no SQLite dependency.

Endpoints:
    POST   /sessions                        Create a new chat session
    GET    /sessions                        List all existing sessions
    PATCH  /sessions/{session_id}/rename    Rename a session
    DELETE /sessions/{session_id}           Delete a session and all its data
    POST   /sessions/{session_id}/chat      Stream a response from the RAG agent
    POST   /rag/mode                        Toggle between RAG and GRAG mode

Environment variables:
    POSTGRES_DSN    Full psycopg3 connection string, e.g.:
                    "postgresql://user:password@host:port/dbname"

Usage:
    python main.py          (recommended on Windows — uses SelectorEventLoop)
    uvicorn main:app        (Linux / macOS only)
"""

# ---------------------------------------------------------------------------
# Windows event loop fix — must be at the very top, before any async imports
# ---------------------------------------------------------------------------
# psycopg3's async mode requires SelectorEventLoop. On Windows, Python 3.8+
# defaults to ProactorEventLoop, which is incompatible. We force the correct
# policy here, before uvicorn or any other library touches the event loop.
import sys
import asyncio
import selectors

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ---------------------------------------------------------------------------
# Standard imports
# ---------------------------------------------------------------------------

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from scripts.utilities.pg_db_utils import (
    list_sessions,
    add_session,
    rename_session,
    delete_session,
)
from scripts.RAG import RAG


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Full psycopg3 DSN — read from environment so credentials are never hardcoded.
# Raises KeyError at startup if unset, which is intentional: a missing DSN
# should be a hard failure, not a silent misconfiguration.
POSTGRES_DSN: str = os.environ["POSTGRES_DSN"]
RAG_DB: str = os.environ["RAG_DB"]

# Adjust to match your deployment
RAG_WORKING_DIR = Path("rag_working_dir")
RAG_PDF_DIR: Path | None = None  # Set to a Path to ingest new PDFs on startup


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------

class AppState:
    """Holds shared application-level resources initialised at startup."""
    pool: AsyncConnectionPool
    rag: RAG


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialise all shared resources on startup and tear them down on shutdown.

    Startup sequence:
        1. Open a shared AsyncConnectionPool.
        2. Create the sessions table if it does not exist.
        3. Construct AsyncPostgresSaver on the same pool and run its migrations
           (creates checkpoints / checkpoint_writes / checkpoint_blobs tables).
        4. Instantiate the RAG agent with the Postgres checkpointer.

    The pool is closed last, after all dependent resources are done with it.

    Note on Windows:
        The SelectorEventLoop is set at module level (above) before this runs.
        Uvicorn must also be launched with loop="none" or via asyncio.run() with
        a SelectorEventLoop — see the __main__ block at the bottom of this file.
    """
    # 1. Open the shared connection pool
    pool = AsyncConnectionPool(POSTGRES_DSN, open=False)
    await pool.open()
    state.pool = pool

    # 2. Ensure the sessions table exists.
    # Done directly on the shared pool rather than via create_pgdb(), which
    # opens its own internal pool — we want a single pool for everything.
    async with pool.connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id    TEXT PRIMARY KEY,
                title TEXT DEFAULT 'Untitled'
            );
        """)

    # 3. Set up AsyncPostgresSaver and run its schema migrations
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()  # Creates LangGraph checkpoint tables if absent

    # 4. Instantiate the RAG agent with the Postgres checkpointer
    state.rag = RAG(
        DB_URI=RAG_DB,
        pdf_docs_dir=None,
        working_dir_path=RAG_WORKING_DIR,
        checkpointer=checkpointer,
    )

    yield

    # Shutdown: close the pool (releases all connections for both session DB
    # and the checkpointer, since they share the same pool)
    await pool.close()


app = FastAPI(
    title="NICE TSD RAG Agent API",
    description="Chat with the NICE TSD Expert Agent — supports RAG and Graph RAG modes.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

async def get_pool() -> AsyncConnectionPool:
    """FastAPI dependency: returns the shared Postgres connection pool."""
    return state.pool


async def get_rag() -> RAG:
    """FastAPI dependency: returns the shared RAG instance."""
    return state.rag


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class CreateSessionRequest(BaseModel):
    title: str = "Untitled"


class RenameSessionRequest(BaseModel):
    title: str


class ChatRequest(BaseModel):
    message: str


class ModeRequest(BaseModel):
    mode: str  # "rag" or "grag"


class SessionResponse(BaseModel):
    session_id: str
    title: str


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# Session endpoints
# ---------------------------------------------------------------------------

@app.post(
    "/sessions",
    response_model=SessionResponse,
    summary="Create a new chat session",
    tags=["Sessions"],
)
async def create_session(
    body: CreateSessionRequest,
    pool: AsyncConnectionPool = Depends(get_pool),
):
    """
    Create a new chat session with a UUID as the thread ID.

    The UUID is stored in the sessions table and used as the LangGraph
    thread ID for all subsequent chat calls on this session.

    Args:
        body: Optional title for the session. Defaults to 'Untitled'.

    Returns:
        The new session's ID and title.
    """
    session_id = str(uuid.uuid4())
    result = await add_session(pool, session_id, body.title)

    if result.startswith("Error"):
        raise HTTPException(status_code=409, detail=result)

    return SessionResponse(session_id=session_id, title=body.title)


@app.get(
    "/sessions",
    response_model=list[SessionResponse],
    summary="List all chat sessions",
    tags=["Sessions"],
)
async def get_sessions(pool: AsyncConnectionPool = Depends(get_pool)):
    """
    Return all existing chat sessions.

    Returns:
        A list of session objects with their IDs and titles.
    """
    rows = await list_sessions(pool)
    return [SessionResponse(session_id=row[0], title=row[1]) for row in rows]


@app.patch(
    "/sessions/{session_id}/rename",
    response_model=MessageResponse,
    summary="Rename a chat session",
    tags=["Sessions"],
)
async def rename_session_endpoint(
    session_id: str,
    body: RenameSessionRequest,
    pool: AsyncConnectionPool = Depends(get_pool),
):
    """
    Update the display title of an existing session.

    Args:
        session_id: The UUID of the session to rename.
        body: The new title.

    Returns:
        A confirmation message.

    Raises:
        404 if no session with the given ID exists.
    """
    result = await rename_session(pool, session_id, body.title)

    if result.startswith("Error"):
        raise HTTPException(status_code=404, detail=result)

    return MessageResponse(message=result)


@app.delete(
    "/sessions/{session_id}",
    response_model=MessageResponse,
    summary="Delete a chat session",
    tags=["Sessions"],
)
async def delete_session_endpoint(
    session_id: str,
    pool: AsyncConnectionPool = Depends(get_pool),
):
    """
    Delete a session and all its associated data.

    Removes the session row and cascades to the LangGraph checkpoint tables
    (checkpoints, checkpoint_writes, checkpoint_blobs) for the same thread ID,
    so no orphaned conversation state is left behind.

    Args:
        session_id: The UUID of the session to delete.

    Returns:
        A confirmation message.

    Raises:
        404 if the session does not exist.
        500 if a database error occurs during deletion.
    """
    result = await delete_session(pool, session_id)

    if result.startswith("Note"):
        raise HTTPException(status_code=404, detail=result)
    if result.startswith("Database error"):
        raise HTTPException(status_code=500, detail=result)

    return MessageResponse(message=result)


# ---------------------------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------------------------

@app.post(
    "/sessions/{session_id}/chat",
    summary="Stream a response from the RAG agent",
    tags=["Chat"],
    response_description="A server-sent event stream of response tokens.",
)
async def chat(
    session_id: str,
    body: ChatRequest,
    pool: AsyncConnectionPool = Depends(get_pool),
    rag: RAG = Depends(get_rag),
):
    """
    Send a message to the RAG agent and stream the response token by token.

    Uses Server-Sent Events (SSE) over `text/event-stream`. Each token is
    yielded as a `data: <token>\\n\\n` frame. The conversation history is
    maintained automatically by LangGraph via AsyncPostgresSaver, keyed on
    the session UUID as the thread ID.

    The session is verified to exist before the stream is opened, so a 404
    is returned as a normal HTTP error rather than appearing mid-stream.

    Args:
        session_id: The UUID of the session (used as the LangGraph thread ID).
        body: The user's message.

    Returns:
        A streaming SSE response terminated with `data: [DONE]\\n\\n`.

    Raises:
        404 if the session does not exist.
    """
    # Verify the session exists before opening the stream
    sessions = await list_sessions(pool)
    if not any(row[0] == session_id for row in sessions):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    async def token_generator():
        async for token in rag.a_analyze_stream(body.message, thread_id=session_id):
            # SSE format: each frame must be "data: <content>\n\n"
            yield f"data: {token}\n\n"
        # Sentinel frame — mirrors OpenAI streaming convention
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Prevents Nginx from batching the stream
        },
    )


# ---------------------------------------------------------------------------
# RAG mode toggle
# ---------------------------------------------------------------------------

@app.post(
    "/rag/mode",
    response_model=MessageResponse,
    summary="Toggle between RAG and GRAG mode",
    tags=["Configuration"],
)
async def set_rag_mode(
    body: ModeRequest,
    rag: RAG = Depends(get_rag),
):
    """
    Switch the RAG agent between standard hybrid retrieval (RAG) and
    graph-augmented retrieval (GRAG) mode.

    Rebuilds the agent's tool list and system prompt in place. All existing
    conversation threads are preserved in Postgres — only future invocations
    are affected by the mode change.

    Args:
        body: `mode` must be either `"rag"` or `"grag"` (case-insensitive).

    Returns:
        A confirmation message indicating the newly active mode.

    Raises:
        400 if an unrecognised mode string is provided.
    """
    mode = body.mode.lower()

    if mode == "rag":
        result = rag.switch_RAG()
    elif mode == "grag":
        result = rag.switch_GRAG()
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode '{body.mode}'. Must be 'rag' or 'grag'."
        )

    return MessageResponse(message=result)


# ---------------------------------------------------------------------------
# Entrypoint — Windows-compatible launcher
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    # On Windows, uvicorn must be told not to install its own event loop policy
    # so that our SelectorEventLoop (set at the top of this file) is preserved.
    # loop="none" tells uvicorn to use whatever loop is already running.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True spawns subprocesses that reset the loop policy
        loop="none",    # Critical on Windows: do not let uvicorn override the loop
    )