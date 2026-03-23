"""
scripts/utilities/metadata_utils.py

Utilities for extracting and correcting PDF metadata using NLP and PyMuPDF (fitz).

Workflow:
    1. For each PDF in a directory, the first page is extracted as lines of text.
    2. `extract_authors` uses NLTK named-entity recognition to identify author lines.
    3. `set_title_author` writes the inferred title (from filename) and authors
       back into the PDF's metadata, saving the result to an output directory.

Dependencies:
    - nltk:  word_tokenize, pos_tag, ne_chunk (requires 'punkt', 'averaged_perceptron_tagger',
             'maxent_ne_chunker', 'words' corpora to be downloaded)
    - fitz:  PyMuPDF, for reading and writing PDF files
    - tqdm:  Progress bar for batch processing
"""

from nltk import word_tokenize, pos_tag, ne_chunk
import re
from pathlib import Path
from tqdm import tqdm
import os
import fitz


def is_person_nltk(text: str) -> bool:
    """
    Check whether a string appears to contain a person's name using NLTK NER.

    Tokenizes the text, applies part-of-speech tagging, and runs the NLTK
    named-entity chunker. Returns True if any chunk is labelled 'PERSON'.

    Args:
        text (str): A short string to classify (e.g. a comma-separated name fragment).

    Returns:
        bool: True if a PERSON entity is detected, False otherwise or on error.
    """
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


def clean_text(text: str) -> str:
    """
    Strip non-alphabetic characters from a string and normalize whitespace.

    Retains letters, spaces, hyphens, and periods (to preserve initials and
    hyphenated names). Collapses any resulting runs of whitespace into single spaces.

    Args:
        text (str): Raw text to clean (e.g. a line extracted from a PDF page).

    Returns:
        str: The cleaned, whitespace-normalized string.
    """
    pattern = r'[^a-zA-Z\s\.\-]'
    cleaned = re.sub(pattern, '', text)
    return " ".join(cleaned.split())


def extract_authors(page_0: list[str]) -> list[str]:
    """
    Infer the list of authors from the first page of a PDF.

    Iterates over lines of text, splitting each on commas and cleaning each piece.
    Uses `is_person_nltk` to score how many pieces look like person names. Lines
    where more than 60% of pieces are identified as names are treated as author lines.

    Stops early once the score drops below the threshold after a high-scoring
    run, on the assumption that author listings appear as a contiguous block
    near the top of the first page.

    Args:
        page_0 (list[str]): Lines of text extracted from the first page of a PDF.

    Returns:
        list[str]: A flat list of cleaned author name strings.
    """
    authors = []
    last_set_score = 0.0

    for line in page_0:
        pieces = line.split(",")
        cleaned_pieces = [clean_text(piece) for piece in pieces]
        cleaned_pieces = [piece for piece in cleaned_pieces if len(piece) > 0]

        if not cleaned_pieces:
            continue

        # Score this line: fraction of pieces that look like person names
        authors_flag = [is_person_nltk(piece) for piece in cleaned_pieces]
        score = authors_flag.count(True) / len(authors_flag)

        if score > 0.6:
            authors += cleaned_pieces

        # Stop once we've passed a high-scoring block and the score drops off
        if last_set_score > 0.6 and score < 0.6:
            break

        last_set_score = score

    return authors


def set_title_author(input_path: Path, output_path: Path) -> None:
    """
    Batch-update PDF metadata by inferring title and authors for each file.

    For each PDF in `input_path`:
        - Sets the 'title' metadata field from the filename (without extension).
        - Sets the 'author' metadata field using `extract_authors` on the first page.
        - Saves the updated PDF to `output_path`, creating it if it doesn't exist.

    Args:
        input_path (Path):  Directory containing the source PDF files.
        output_path (Path): Directory to write the updated PDF files to.
                            Created automatically if it does not exist.

    Returns:
        None
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"Created output folder: {output_path}")
    else:
        print(f"Output folder already exists: {output_path}")

    print("Inferring titles from filenames and authors from first pages...")

    for filename in tqdm(os.listdir(input_path)):
        file_path = input_path / filename
        doc = fitz.open(file_path)

        page_0_lines = doc.load_page(0).get_text().split('\n')  # type: ignore
        authors = extract_authors(page_0_lines)

        new_metadata = {
            "title": filename.split('.pdf')[0],
            "author": ", ".join(authors),
        }

        doc.set_metadata(new_metadata)
        doc.save(output_path / filename)
        doc.close()

    print("Metadata update complete.")