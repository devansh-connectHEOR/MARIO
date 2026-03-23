"""
scripts/utilities/db_utils.py

Synchronous utility functions for managing a SQLite database of chat sessions.
Uses the standard sqlite3 library for database operations.

This is the synchronous counterpart to async_db_utils.py, intended for use
in non-async contexts (e.g. app initialization, CLI tools, or testing).

Tables managed:
    - sessions:     Stores session metadata (id, title)
    - checkpoints:  Managed externally; cleaned up on session deletion
    - writes:       Managed externally; cleaned up on session deletion
"""

import sqlite3
from pathlib import Path


def create_sqldb(name: str, path: Path) -> sqlite3.Connection:
    """
    Create (or connect to) a SQLite database and initialize the sessions table.

    `check_same_thread=False` is set to allow the connection to be used
    across threads, which is required in multi-threaded environments (e.g. web servers).

    Args:
        name (str): The filename of the database (e.g. "sessions.db").
        path (Path): The directory in which to create the database file.

    Returns:
        sqlite3.Connection: An open connection to the database.
    """
    sqldb = sqlite3.connect(path / name, check_same_thread=False)

    create_table_query = """
    CREATE TABLE IF NOT EXISTS sessions (
        id    TEXT PRIMARY KEY,
        title TEXT DEFAULT 'Untitled'
    );
    """

    cursor = sqldb.cursor()
    cursor.execute(create_table_query)
    sqldb.commit()
    return sqldb


def list_sessions(sqldb: sqlite3.Connection) -> list[tuple]:
    """
    Retrieve all sessions from the database.

    Args:
        sqldb (sqlite3.Connection): An open database connection.

    Returns:
        list[tuple]: A list of (id, title) rows, or an empty list if none exist.
    """
    cursor = sqldb.cursor()
    cursor.execute("SELECT * FROM sessions;")
    return cursor.fetchall()


def add_session(
    sqldb: sqlite3.Connection, id: str, title: str = "Untitled"
) -> str:
    """
    Insert a new session into the database.

    Args:
        sqldb (sqlite3.Connection): An open database connection.
        id (str): A unique identifier for the session.
        title (str): A human-readable name for the session. Defaults to 'Untitled'.

    Returns:
        str: A success message, or an error message if the ID already exists
             or an unexpected error occurs.
    """
    query = "INSERT INTO sessions (id, title) VALUES (?, ?)"
    cursor = sqldb.cursor()

    try:
        cursor.execute(query, (id, title))
        sqldb.commit()
        return f"Success: Session '{title}' added."

    except sqlite3.IntegrityError:
        # Triggers when 'id' violates the PRIMARY KEY uniqueness constraint
        return f"Error: A session with ID '{id}' already exists."

    except Exception as e:
        return f"An unexpected error occurred: {e}"


def rename_session(sqldb: sqlite3.Connection, id: str, title: str) -> str:
    """
    Update the title of an existing session.

    Args:
        sqldb (sqlite3.Connection): An open database connection.
        id (str): The unique identifier of the session to rename.
        title (str): The new title for the session.

    Returns:
        str: A success message, or a message indicating no session was found.
    """
    query = "UPDATE sessions SET title = ? WHERE id = ?"
    cursor = sqldb.cursor()
    cursor.execute(query, (title, id))

    if cursor.rowcount == 0:
        return f"Error: No session found with ID '{id}'."

    sqldb.commit()
    return f"Success: Session '{id}' renamed to '{title}'."


def delete_session(sqldb: sqlite3.Connection, session_id: str) -> str:
    """
    Delete a session and all its associated data across related tables.

    Removes records from the 'sessions', 'checkpoints', and 'writes' tables
    that match the given session ID. The operation is atomic — if any step
    fails, all changes are rolled back.

    Args:
        sqldb (sqlite3.Connection): An open database connection.
        session_id (str): The unique identifier of the session to delete.

    Returns:
        str: A success message, a note if no records were found,
             or an error message with rollback confirmation if a DB error occurs.
    """
    cursor = sqldb.cursor()

    try:
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))

        # Commit only if all deletions succeeded
        sqldb.commit()

        if cursor.rowcount > 0:
            return f"Success: Session '{session_id}' removed from all tables."
        else:
            return f"Note: No matching records found for session ID '{session_id}'."

    except sqlite3.Error as e:
        # Undo any partial deletions if something went wrong
        sqldb.rollback()
        return f"Database error: {e}. Changes rolled back."