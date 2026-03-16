import aiosqlite
from pathlib import Path

async def create_sqldb(name: str, path: Path):
    sqldb = await aiosqlite.connect(path / name)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT DEFAULT 'Untitled'
    );
    """
    async with sqldb.execute(create_table_query):
        pass
    await sqldb.commit()
    return sqldb

async def list_sessions(sqldb):
    async with sqldb.execute("SELECT * FROM sessions;") as cursor:
        return await cursor.fetchall()

async def add_session(sqldb, id, title="Untitled"):
    query = "INSERT INTO sessions (id, title) VALUES (?, ?)"
    try:
        await sqldb.execute(query, (id, title))
        await sqldb.commit()
        return f"Success: Session {title} added."

    except aiosqlite.IntegrityError:
        return f"Error: A session with ID '{id}' already exists."

    except Exception as e:
        return f"An unexpected error occurred: {e}"

async def rename_session(sqldb, id, title):
    query = "UPDATE sessions SET title = ? WHERE id = ?"
    async with sqldb.execute(query, (title, id)) as cursor:
        if cursor.rowcount == 0:
            return f"No session found with ID: {id}"
    await sqldb.commit()
    return "Title updated successfully!"

async def delete_sessions(sqldb, session_id):
    try:
        await sqldb.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await sqldb.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))
        async with sqldb.execute(
            "DELETE FROM writes WHERE thread_id = ?", (session_id,)
        ) as cursor:
            rowcount = cursor.rowcount

        await sqldb.commit()

        if rowcount > 0:
            return f"Success: Session {session_id} removed from all tables."
        else:
            return "Note: No matching records found to delete."

    except aiosqlite.Error as e:
        await sqldb.rollback()
        return f"Database error: {e}. Changes rolled back."