from nltk import word_tokenize, pos_tag, ne_chunk
import re
from pathlib import Path
from tqdm import tqdm
import os
import fitz
import sqlite3

#---Metadata Correction Functions---
#
def is_person_nltk(text):
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        chunks = ne_chunk(tagged)
        
        for chunk in chunks:
            if hasattr(chunk, 'label') and chunk.label() == 'PERSON':
                return True
        return False
    except Exception:
        return False

def clean_text(text):
    pattern = r'[^a-zA-Z\s\.\-]'
    cleaned = re.sub(pattern, '', text)
    return " ".join(cleaned.split())

def extract_authors(page_0: list[str]) -> list[str]:
    authors = []
    authors_flag = []
    last_set_score = 0.0
    
    for line in page_0:
        pieces = line.split(",")
        cleaned_pieces = [clean_text(piece) for piece in pieces]

        cleaned_pieces = [piece for piece in cleaned_pieces if len(piece) > 0]
        
        if len(cleaned_pieces) == 0:
            continue
        
        for piece in cleaned_pieces:
            authors_flag.append(is_person_nltk(piece))
        
        score = authors_flag.count(True) / len(authors_flag)
        
        if score > 0.6 and len(authors_flag) > 0:
            authors += cleaned_pieces
        
        if last_set_score > 0.6 and score < 0.6:
            break
        
        last_set_score = score
        authors_flag = []

    return authors

def set_title_author(input_path: Path, output_path: Path):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"Created folder: {output_path}")
    else: print(f"Output folder exists")
    print("Extracting title from the file name and authors from the first page and setting them as meta data")
    for filename in tqdm(os.listdir(input_path)):
        file_path = input_path / filename
        doc = fitz.open(file_path)
        page_0 = doc.load_page(0).get_text().split('\n')
        authors = extract_authors(page_0)
        new_metadata = {
            "title": filename.split('.pdf')[0],
            'author': ",".join(authors)
        }
        doc.set_metadata(new_metadata)
        doc.save(output_path / filename)
        doc.close()
    print("Updating metadata completed")

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
        return "Success: Session added."
    
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