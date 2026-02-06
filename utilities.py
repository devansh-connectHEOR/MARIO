from nltk import word_tokenize, pos_tag, ne_chunk
import re
from pathlib import Path
from tqdm import tqdm
import os
import fitz

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

def set_title_author(input: Path, output: Path) -> str:
    if not os.path.exists(output):
        os.makedir(output)
        print(f"Created folder: {output}")
    else: print(f"Output folder exists")
    print("Extracting title from the file name and authors from the first page and setting them as meta data")
    for filename in tqdm(os.listdir(input)):
        file_path = input / filename
        doc = fitz.open(file_path)
        page_0 = doc.load_page(0).get_text().split('\n')
        authors = extract_authors(page_0)
        new_metadata = {
            "title": filename.split('.pdf')[0],
            'authors': ",".join(authors)
        }
        doc.set_metadata(new_metadata)
        doc.save(output / filename)
        doc.close()
    print("Updating metadata completed")

