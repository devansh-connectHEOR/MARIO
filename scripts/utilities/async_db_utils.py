"""
scripts/utilities/async_db_utils.py

Async utility functions for managing a SQLite database of chat sessions.
Uses aiosqlite for non-blocking database operations.

Tables managed:
    - sessions:     Stores session metadata (id, title)
    - checkpoints:  Managed externally; cleaned up on session deletion
    - writes:       Managed externally; cleaned up on session deletion
"""

import aiosqlite
from pathlib import Path

from aiosqlite import Row
from typing import Iterable


async def create_sqldb(name: str, path: Path) -> aiosqlite.Connection:
    """
    Create (or connect to) a SQLite database and initialize the sessions table.

    Args:
        name (str): The filename of the database (e.g. "sessions.db").
        path (Path): The directory in which to create the database file.

    Returns:
        aiosqlite.Connection: An open connection to the database.
    """
    sqldb = await aiosqlite.connect(path / name)

    create_table_query = """
    CREATE TABLE IF NOT EXISTS sessions (
        id    TEXT PRIMARY KEY,
        title TEXT DEFAULT 'Untitled'
    );
    """

    async with sqldb.execute(create_table_query):
        pass

    await sqldb.commit()
    return sqldb


async def list_sessions(sqldb: aiosqlite.Connection) -> Iterable[Row]:
    """
    Retrieve all sessions from the database.

    Args:
        sqldb (aiosqlite.Connection): An open database connection.

    Returns:
        list[tuple]: A list of (id, title) rows, or an empty list if none exist.
    """
    async with sqldb.execute("SELECT * FROM sessions;") as cursor:
        return await cursor.fetchall()


async def add_session(
    sqldb: aiosqlite.Connection, id: str, title: str = "Untitled"
) -> str:
    """
    Insert a new session into the database.

    Args:
        sqldb (aiosqlite.Connection): An open database connection.
        id (str): A unique identifier for the session.
        title (str): A human-readable name for the session. Defaults to 'Untitled'.

    Returns:
        str: A success message, or an error message if the ID already exists
             or an unexpected error occurs.
    """
    query = "INSERT INTO sessions (id, title) VALUES (?, ?)"

    try:
        await sqldb.execute(query, (id, title))
        await sqldb.commit()
        return f"Success: Session '{title}' added."

    except aiosqlite.IntegrityError:
        return f"Error: A session with ID '{id}' already exists."

    except Exception as e:
        return f"An unexpected error occurred: {e}"


async def rename_session(sqldb: aiosqlite.Connection, id: str, title: str) -> str:
    """
    Update the title of an existing session.

    Args:
        sqldb (aiosqlite.Connection): An open database connection.
        id (str): The unique identifier of the session to rename.
        title (str): The new title for the session.

    Returns:
        str: A success message, or a message indicating no session was found.
    """
    query = "UPDATE sessions SET title = ? WHERE id = ?"

    async with sqldb.execute(query, (title, id)) as cursor:
        if cursor.rowcount == 0:
            return f"Error: No session found with ID '{id}'."

    await sqldb.commit()
    return f"Success: Session '{id}' renamed to '{title}'."


async def delete_session(sqldb: aiosqlite.Connection, session_id: str) -> str:
    """
    Delete a session and all its associated data across related tables.

    Removes records from the 'sessions', 'checkpoints', and 'writes' tables
    that match the given session ID. The operation is atomic — if any step
    fails, all changes are rolled back.

    Args:
        sqldb (aiosqlite.Connection): An open database connection.
        session_id (str): The unique identifier of the session to delete.

    Returns:
        str: A success message, a note if no records were found,
             or an error message with rollback confirmation if a DB error occurs.
    """
    try:
        await sqldb.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await sqldb.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?", (session_id,)
        )
        async with sqldb.execute(
            "DELETE FROM writes WHERE thread_id = ?", (session_id,)
        ) as cursor:
            rowcount = cursor.rowcount

        await sqldb.commit()

        if rowcount > 0:
            return f"Success: Session '{session_id}' removed from all tables."
        else:
            return f"Note: No matching records found for session ID '{session_id}'."

    except aiosqlite.Error as e:
        await sqldb.rollback()
        return f"Database error: {e}. Changes rolled back."