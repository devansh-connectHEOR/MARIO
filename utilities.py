from nltk import word_tokenize, pos_tag, ne_chunk

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

import re

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