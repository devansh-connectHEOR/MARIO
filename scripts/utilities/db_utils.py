import sqlite3
from pathlib import Path

def create_sqldb(name:str, path: Path):
    sqldb = sqlite3.connect(path / name, check_same_thread=False)
    create_table_query = """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        title TEXT DEFAULT 'Untitled'
    );
    """
    cursor = sqldb.cursor()
    cursor.execute(create_table_query)
    sqldb.commit()
    return sqldb

def list_sessions(sqldb):
    query = "SELECT * FROM sessions;"
    cursor = sqldb.cursor()
    cursor.execute(query)
    return cursor.fetchall()

def add_session(sqldb, id, title = "Untitled"):
    query = "INSERT INTO sessions (id, title) VALUES (?, ?)"
    cursor = sqldb.cursor()
    try:
        cursor.execute(query, (id, title))
        sqldb.commit()
        return f"Success: Session {title} added."
    
    except sqlite3.IntegrityError:
        # This triggers if 'id' is a Primary Key and already exists
        return f"Error: A session with ID '{id}' already exists."
    
    except Exception as e:
        # Catch-all for other potential database errors
        return f"An unexpected error occurred: {e}"

def rename_session(sqldb, id, title):
    query = "UPDATE sessions SET title = ? WHERE id = ?"
    cursor = sqldb.cursor()  

    cursor.execute(query, (title, id))

    # 3. Check if any row was actually changed
    if cursor.rowcount == 0:
        return f"No session found with ID: {id}"
    else:
        sqldb.commit()
        return "Title updated successfully!"

def delete_sessions(sqldb, session_id):
    cursor = sqldb.cursor()
    
    try:
        
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (session_id,))

        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (session_id,))
        
        # 3. Commit only if both commands succeeded
        sqldb.commit()
        
        if cursor.rowcount > 0:
            return f"Success: Session {session_id} removed from all tables."
        else:
            return "Note: No matching records found to delete."
            
    except sqlite3.Error as e:
        # If anything goes wrong, undo any partial deletions
        sqldb.rollback()
        return f"Database error: {e}. Changes rolled back."