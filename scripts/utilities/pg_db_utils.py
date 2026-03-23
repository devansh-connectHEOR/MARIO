"""
scripts/utilities/pg_db_utils.py

Async utility functions for managing a PostgreSQL database of chat sessions.
Uses psycopg3 (psycopg) with an AsyncConnectionPool for efficient connection reuse.

This is the PostgreSQL counterpart to async_db_utils.py (SQLite). Key differences:
    - Connections are managed via a pool rather than a single connection object.
    - Uses %s placeholders instead of SQLite's ? placeholders.
    - Transactions are handled explicitly via conn.transaction().
    - psycopg3 auto-rolls back on context manager exit if an exception is raised,
      so manual rollback() calls are not needed.

Tables managed:
    - sessions:           Stores session metadata (id, title)
    - checkpoints:        Managed externally; cleaned up on session deletion
    - checkpoint_writes:  Managed externally; cleaned up on session deletion
    - checkpoint_blobs:   Managed externally; cleaned up on session deletion
"""

import psycopg
from psycopg_pool import AsyncConnectionPool


async def create_pgdb(dsn: str) -> AsyncConnectionPool:
    """
    Create an async connection pool and initialize the sessions table.

    Opens the pool explicitly (open=False defers opening until await pool.open())
    which allows the pool to be safely created at import time or before an event
    loop is running.

    Args:
        dsn (str): A PostgreSQL connection string, e.g.
                   "postgresql://user:password@host:port/dbname".

    Returns:
        AsyncConnectionPool: An open pool ready for use.
    """
    pool = AsyncConnectionPool(dsn, open=False)
    await pool.open()

    async with pool.connection() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id    TEXT PRIMARY KEY,
                title TEXT DEFAULT 'Untitled'
            );
        """)

    return pool # type: ignore


async def list_sessions(pool: AsyncConnectionPool) -> list[tuple]:
    """
    Retrieve all sessions from the database.

    Args:
        pool (AsyncConnectionPool): An open connection pool.

    Returns:
        list[tuple]: A list of (id, title) rows, or an empty list if none exist.
    """
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT * FROM sessions;")
            return await cur.fetchall()


async def add_session(
    pool: AsyncConnectionPool, id: str, title: str = "Untitled"
) -> str:
    """
    Insert a new session into the database.

    Performs a pre-flight SELECT to check for ID conflicts before inserting,
    returning a clear error message rather than raising an exception if the
    ID already exists.

    Args:
        pool (AsyncConnectionPool): An open connection pool.
        id (str): A unique identifier for the session.
        title (str): A human-readable name for the session. Defaults to 'Untitled'.

    Returns:
        str: A success message, or an error message if the ID already exists
             or an unexpected error occurs.
    """
    check_query = "SELECT id FROM sessions WHERE id = %s"
    insert_query = "INSERT INTO sessions (id, title) VALUES (%s, %s)"

    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(check_query, (id,))
                if await cur.fetchone():
                    return f"Error: A session with ID '{id}' already exists."

            await conn.execute(insert_query, (id, title))

        return f"Success: Session '{title}' added."

    except Exception as e:
        return f"An unexpected error occurred: {e}"


async def rename_session(
    pool: AsyncConnectionPool, id: str, title: str
) -> str:
    """
    Update the title of an existing session.

    Args:
        pool (AsyncConnectionPool): An open connection pool.
        id (str): The unique identifier of the session to rename.
        title (str): The new title for the session.

    Returns:
        str: A success message, or a message indicating no session was found.
    """
    query = "UPDATE sessions SET title = %s WHERE id = %s"

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, (title, id))
            if cur.rowcount == 0:
                return f"Error: No session found with ID '{id}'."

    return f"Success: Session '{id}' renamed to '{title}'."


async def delete_session(pool: AsyncConnectionPool, session_id: str) -> str:
    """
    Delete a session and all its associated data across related tables.

    Wraps all deletions in an explicit transaction via conn.transaction(), so
    either all deletions succeed or none are committed. psycopg3 automatically
    rolls back the transaction if an exception propagates out of the block.

    Args:
        pool (AsyncConnectionPool): An open connection pool.
        session_id (str): The unique identifier of the session to delete.

    Returns:
        str: A success message, a note if no session was found,
             or an error message if a database error occurs.
    """
    try:
        async with pool.connection() as conn:
            async with conn.transaction():
                async with conn.cursor() as cur:
                    await cur.execute(
                        "DELETE FROM sessions WHERE id = %s", (session_id,)
                    )
                    rowcount = cur.rowcount

                await conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = %s", (session_id,)
                )
                await conn.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id = %s", (session_id,)
                )
                await conn.execute(
                    "DELETE FROM checkpoint_blobs WHERE thread_id = %s", (session_id,)
                )

        if rowcount > 0:
            return f"Success: Session '{session_id}' removed from all tables."
        else:
            return f"Note: No session found with ID '{session_id}'."

    except psycopg.Error as e:
        # psycopg3 rolls back automatically on context manager exit;
        # this block handles logging/reporting of the error.
        return f"Database error: {e}. Changes rolled back."