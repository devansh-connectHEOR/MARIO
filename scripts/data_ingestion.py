# Making imports
from pathlib import Path
import os
from tqdm import tqdm
import re

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"     # Need to be disabled because of work laptop restrictions

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from docling.document_converter import ConversionResult
import fitz
from PIL import Image

pipeline_options = PdfPipelineOptions()

pipeline_options.do_formula_enrichment = True
pipeline_options.generate_page_images = True

default_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def read_documents_from_directory(input_path: Path, converter: DocumentConverter = default_converter) -> list[tuple[ConversionResult, str]]:
        print(f"Reading files from directory {input}")
        dox = []   
        i = 0
        with os.scandir(input_path) as entries:
            pbar = tqdm(entries, dynamic_ncols=True, unit="doc", desc="Loading docs", leave=True)
            for entry in pbar:
                if entry.is_file():
                    pbar.set_description(f"Working on {entry}")
                    file_path = input / entry
                    d = fitz(file_path)
                    meta = d.metadata
                    metadata = f"{meta['title']} | {meta['authors']} | TSD"
                    doc = (converter.convert(file_path), metadata)
                    dox.append(doc)
                    i+=1
        print(f"Read {i} documents")
        return dox

def read_documents_from_list(input_paths: list[Path], converter: DocumentConverter = default_converter) -> list[tuple[ConversionResult, str]]
        print(f"Reading files")
        dox = []
        i = 0
        pbar = tqdm(input_paths, dynamic_ncols=True, unit="doc", desc="Loading docs", leave=True)
        for entry in pbar:
            if entry.is_file():
                pbar.set_description(f"Working on {entry}")
                file_path = input / entry
                d = fitz(file_path)
                meta = d.metadata
                metadata = f"{meta['title']} | {meta['authors']} | TSD"
                doc = (converter.convert(file_path), metadata)
                dox.append(doc)
                i+=1
        print(f"Read {i} documents")
        return dox

def read_documents(input_path: Path | list[Path], converter: DocumentConverter = default_converter) -> list[tuple[ConversionResult, str]]:
        if isinstance(input_path, Path):
            return read_documents_from_directory(input_path, converter)
        
        elif isinstance(input_path, list):
            return read_documents_from_list(input_path, converter)
        
        else: 
            raise TypeError(f"Expected path to a directory or list of paths to PDFs and DoclingConverter, got {type(input_path).__name__} and {type(converter).__name__}")

def extract_markdown_images(docs: list[(ConversionResult, str)], 
                            markdown_path: Path, 
                            images_path: Path) -> bool:
        img_metadata = ["Image Path | Document | Document Type | Page Number | Caption"]
        doc_metadata = ["Document Title | Authors | Document Type"]

        pbar = tqdm(docs, dynamic_ncols=True, unit="doc", desc="Extracting data", leave=True)
        for doc, meta in pbar:
             title = meta.split(" | ")[0]
             doc_type = meta.split(" | ")[2]
             mkd = doc.document.export_to_markdown()
             output_file = markdown_path / f"{title}.md"
             output_file.write_text(mkd, encoding="utf-8")
             doc_metadata += meta

             for i, img in enumerate(doc.document.pictures):
                i += 1
                image = img.get_image(doc.document)
                image_filename = images_path / f"{title}_image_{i}.png"
                image.save(image_filename)
                if img.caption_text(doc.document):  # If caption captured
                     caption = img.caption_text(doc.document)
                else:   # Search for caption
                     pattern = re.compile(rf"^Figure\s*{i}[.:\- ]*", re.IGNORECASE)
                     for item, _ in doc.document.iterate_items():
                          if hasattr(item, 'text'):
                            text = item.text.strip()
                            if pattern.match(text):
                                caption = text
                                break
                if not caption: caption = "Caption not found"
                img_metadata.append(f"{image_filename} | {title} | {doc_type} | {img.prov[0].page_no} | {caption}")
        with open(markdown_path / "metadata.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(doc_metadata))

        with open(images_path / "metadata.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(img_metadata))

