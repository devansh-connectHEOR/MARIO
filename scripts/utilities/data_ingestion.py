"""
scripts/utilities/data_ingestion.py

Utilities for ingesting PDF documents into a multimodal RAG pipeline.

Pipeline overview:
    1. Read:     Load PDFs from a directory or list of paths using Docling,
                 which handles OCR, formula enrichment, and structured extraction.
    2. Extract:  Export each document to Markdown and extract embedded images,
                 saving both to disk alongside a metadata index.
    3. Load:     Read the saved Markdown and image files back as LangChain Documents,
                 optionally splitting text chunks with a TextSplitter.

Environment:
    HF_HUB_DISABLE_SYMLINKS=1 is set at import time to avoid symlink errors
    on systems with restricted permissions (e.g. managed work laptops).
"""

from pathlib import Path
import os
from tqdm import tqdm
import re
import base64
import scripts.utilities.data_ingestion as di

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"  # Required on managed laptops that restrict symlinks

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, ConversionResult
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrAutoOptions
from docling.document_converter import PdfFormatOption

import fitz
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter


# --- Docling Pipeline Configuration ---

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.ocr_options = OcrAutoOptions()
pipeline_options.do_formula_enrichment = True
pipeline_options.generate_page_images = False  # Page images not needed; individual figures are extracted separately

default_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)


# --- Document Reading ---

def read_documents_from_directory(
    input_path: Path,
    converter: DocumentConverter = default_converter
) -> list[tuple[ConversionResult, str]]:
    """
    Recursively find and read all PDF files in a directory.

    Args:
        input_path (Path): Root directory to search for PDF files.
        converter (DocumentConverter): Docling converter to use. Defaults to `default_converter`.

    Returns:
        list[tuple[ConversionResult, str]]: A list of (conversion result, metadata string) tuples.
    """
    all_pdf_files = list(input_path.rglob("*.pdf"))
    return read_documents_from_list(all_pdf_files, converter)


def read_documents_from_list(
    input_paths: list[Path],
    converter: DocumentConverter = default_converter
) -> list[tuple[ConversionResult, str]]:
    """
    Read and convert a list of PDF files using Docling.

    Reads each file's existing PyMuPDF metadata (title, author) and pairs it
    with the Docling conversion result as a pipe-delimited metadata string.

    Args:
        input_paths (list[Path]): Paths to individual PDF files.
        converter (DocumentConverter): Docling converter to use. Defaults to `default_converter`.

    Returns:
        list[tuple[ConversionResult, str]]: A list of (conversion result, metadata string) tuples.
                                            Metadata format: "title | author | TSD"
    """
    print("Reading files...")
    dox = []
    pbar = tqdm(input_paths, dynamic_ncols=True, unit="doc", desc="Loading docs", leave=True)

    for entry in pbar:
        if entry.is_file():
            d = fitz.open(entry)
            meta = d.metadata
            metadata = f"{meta['title']} | {meta['author']} | TSD" # type: ignore
            dox.append((converter.convert(entry), metadata))

    print(f"Read {len(dox)} documents.")
    return dox


def read_documents(
    input_path: Path | list[Path],
    converter: DocumentConverter = default_converter
) -> list[tuple[ConversionResult, str]]:
    """
    Dispatch PDF reading to the appropriate function based on input type.

    Accepts either a directory Path (reads all PDFs recursively) or a list
    of Paths (reads each file directly).

    Args:
        input_path (Path | list[Path]): A directory to search, or a list of PDF file paths.
        converter (DocumentConverter): Docling converter to use. Defaults to `default_converter`.

    Returns:
        list[tuple[ConversionResult, str]]: A list of (conversion result, metadata string) tuples.

    Raises:
        TypeError: If `input_path` is neither a Path nor a list of Paths.
    """
    if isinstance(input_path, Path):
        return read_documents_from_directory(input_path, converter)
    elif isinstance(input_path, list):
        return read_documents_from_list(input_path, converter)
    else:
        raise TypeError(
            f"Expected a Path to a directory or a list of Paths to PDF files, "
            f"got {type(input_path).__name__}."
        )


# --- Extraction ---

def extract_markdown_images(
    docs: list[tuple[ConversionResult, str]],
    markdown_path: Path,
    images_path: Path
) -> None:
    """
    Export each document to Markdown and extract embedded images to disk.

    For each document:
        - Exports the full text as a Markdown file named after the document title.
        - Saves each embedded figure as a PNG, attempting to find its caption
          either from Docling's caption extraction or by scanning the document
          text for a matching "Figure N" pattern.
        - Writes a metadata index (metadata.txt) to both output directories.

    Metadata index format:
        markdown_path/metadata.txt:  "Document | Authors | Document Type"
        images_path/metadata.txt:    "Image Name | Document | Document Type | Page Number | Caption"

    Args:
        docs (list[tuple[ConversionResult, str]]): Output from `read_documents`.
        markdown_path (Path): Directory to write Markdown files and text metadata.
        images_path (Path): Directory to write PNG images and image metadata.

    Returns:
        None
    """
    img_metadata = ["Image Name | Document | Document Type | Page Number | Caption"]
    doc_metadata = ["Document | Authors | Document Type"]

    pbar = tqdm(docs, dynamic_ncols=True, unit="doc", desc="Extracting data", leave=True)

    for doc, meta in pbar:
        title, _, doc_type = meta.split(" | ")

        # Export full document text to Markdown
        mkd = doc.document.export_to_markdown()
        output_file = markdown_path / f"{title}.md"
        output_file.write_text(mkd, encoding="utf-8")
        doc_metadata.append(meta)

        # Extract and save each embedded figure
        for i, img in enumerate(doc.document.pictures, start=1):
            image = img.get_image(doc.document)
            image_filename = images_path / f"{title}_image_{i}.png"
            image.save(image_filename)

            # Attempt 1: use Docling's built-in caption extraction
            caption = img.caption_text(doc.document)

            # Attempt 2: scan document items for a matching "Figure N" label
            if not caption:
                pattern = re.compile(rf"^Figure\s*{i}[.:\- ]*", re.IGNORECASE)
                for item, _ in doc.document.iterate_items():
                    if hasattr(item, 'text'):
                        text = item.text.strip()
                        if pattern.match(text):
                            caption = text
                            break

            if not caption:
                caption = "Caption not found"

            img_metadata.append(
                f"{image_filename.stem} | {title} | {doc_type} | {img.prov[0].page_no} | {caption}"
            )

    with open(markdown_path / "metadata.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(doc_metadata))

    with open(images_path / "metadata.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(img_metadata))

    print(f"Markdown saved to '{markdown_path}', images saved to '{images_path}'.")


# --- Loading ---

def load_data(
    markdown_path: Path,
    images_path: Path,
    splitter: TextSplitter | None = None
) -> tuple[list[Document], list[Document], dict]:
    """
    Load previously extracted Markdown and image files as LangChain Documents.

    Reads the metadata index from each directory, then loads each file's content.
    Text documents are optionally split using `splitter`. Images are base64-encoded
    and stored in a separate dict keyed by image stem.

    Args:
        markdown_path (Path | None): Directory containing Markdown files and metadata.txt.
                                     Pass None to skip text loading.
        images_path (Path | None):   Directory containing PNG files and metadata.txt.
                                     Pass None to skip image loading.
        splitter (TextSplitter | None): Optional text splitter for chunking Markdown content.
                                        If None, each file is loaded as a single Document.

    Returns:
        tuple[list[Document], list[Document], dict]:
            - mkd_docs: LangChain Documents from Markdown files (possibly chunked).
            - img_docs: LangChain Documents from image captions with image metadata.
            - imgs:     Dict mapping image stem (str) to base64-encoded PNG (str).
    """
    mkd_docs = []
    img_docs = []
    imgs = {}

    if markdown_path:
        with open(markdown_path / "metadata.txt", mode='r', encoding='utf-8') as f:
            raw = f.read().split('\n')[1:]  # Skip header row
            metadata = {
                row[0]: {
                    "type": "text",
                    "document": row[0],
                    "authors": row[1],
                    "doc_type": row[2]
                }
                for row in (line.split(' | ') for line in raw)
            }

        for mkd in tqdm(markdown_path.glob("*.md"), dynamic_ncols=True, unit="doc", desc="Loading docs", leave=True):
            with open(mkd, mode='r', encoding='utf-8') as f:
                content = f.read()

            docs = splitter.split_text(content) if splitter else [Document(content)]
            for doc in docs:
                doc.metadata.update(metadata[mkd.stem]) 
            mkd_docs += docs

    if images_path:
        with open(images_path / "metadata.txt", mode='r', encoding='utf-8') as f:
            raw = f.read().split('\n')[1:]  # Skip header row
            metadata = {
                row[0]: {
                    "type": "image",
                    "image": row[0],
                    "document": row[1],
                    "doc_type": row[2],
                    "page_no": row[3],
                    "caption": row[4]
                }
                for row in (line.split(' | ') for line in raw)
            }

        for img in tqdm(images_path.glob("*.png"), dynamic_ncols=True, unit="doc", desc="Loading images", leave=True):
            with open(img, 'rb') as image_file:
                imgs[img.stem] = base64.b64encode(image_file.read()).decode("utf-8")

            img_docs.append(Document(
                page_content=metadata[img.stem]['caption'],
                metadata=metadata[img.stem]
            ))

    return mkd_docs, img_docs, imgs



# --- Export ---

def documents_to_kg_text(documents: list[Document], output_dir: Path) -> None:
    """
    Write each LangChain Document to a plain text file for use in a knowledge graph pipeline.

    Text documents include their full metadata header and page content.
    Image documents include a figure metadata header and the image caption.
    Files are named using the document title (or source document name) plus an
    index suffix to avoid collisions between same-titled documents.

    Args:
        documents (list[Document]): LangChain Documents to export (text or image type).
        output_dir (Path): Directory to write the output .txt files to.

    Returns:
        None
    """
    for i, doc in enumerate(documents):
        if doc.metadata.get("title"):
            # Text document
            name = doc.metadata["title"]
            header = str(doc.metadata)
            body = "Page Content:\n" + str(doc.page_content)
        else:
            # Image document
            name = doc.metadata.get("document", f"unknown_{i}")
            page = doc.metadata.get('page_no')
            header = f"Figure from {name}, Page: {page}"
            body = "Figure Caption:\n" + str(doc.page_content)

        output_file = output_dir / f"{name}{i}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(header + "\n" + body)

    print("Export complete.")