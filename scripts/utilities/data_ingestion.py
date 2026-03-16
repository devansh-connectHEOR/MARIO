# Making imports
from pathlib import Path
import os
from tqdm import tqdm
import re

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"     # Need to be disabled because of work laptop restrictions

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions, OcrAutoOptions
from docling.document_converter import PdfFormatOption
from docling.document_converter import ConversionResult
import fitz
from PIL import Image
import base64

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters.base import TextSplitter

pipeline_options = PdfPipelineOptions()

pipeline_options.do_ocr = True
pipeline_options.ocr_options = OcrAutoOptions()

pipeline_options.do_formula_enrichment = True
pipeline_options.generate_page_images = False

default_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

def read_documents_from_directory(input_path: Path, converter: DocumentConverter = default_converter) -> list[tuple[ConversionResult, str]]:
        all_pdf_files = list(input_path.rglob("*.pdf"))
        dox = read_documents_from_list(all_pdf_files, converter)
        return dox

def read_documents_from_list(input_paths: list[Path], converter: DocumentConverter = default_converter) -> list[tuple[ConversionResult, str]]:
        print(f"Reading files")
        dox = []
        i = 0
        pbar = tqdm(input_paths, dynamic_ncols=True, unit="doc", desc="Loading docs", leave=True)
        for entry in pbar:
            if entry.is_file():
                file_path = entry
                d = fitz.open(file_path)
                meta = d.metadata
                metadata = f"{meta['title']} | {meta['author']} | TSD"
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
                            images_path: Path):
        img_metadata = ["Image Name | Document | Document Type | Page Number | Caption"]
        doc_metadata = ["Document Title | Authors | Document Type"]

        pbar = tqdm(docs, dynamic_ncols=True, unit="doc", desc="Extracting data", leave=True)
        for doc, meta in pbar:
             title = meta.split(" | ")[0]
             doc_type = meta.split(" | ")[2]
             mkd = doc.document.export_to_markdown()
             output_file = markdown_path / f"{title}.md"
             output_file.write_text(mkd, encoding="utf-8")
             doc_metadata.append(meta)

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
                img_metadata.append(f"{image_filename.stem} | {title} | {doc_type} | {img.prov[0].page_no} | {caption}")
        with open(markdown_path / "metadata.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(doc_metadata))

        with open(images_path / "metadata.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(img_metadata))
        
        print(f"Extracted markdown saved to {markdown_path} and images saved to {images_path}")

def load_data(markdown_path: Path = None, images_path: Path = None, splitter: TextSplitter = None) -> list[Document, Document, dict]:
    mkd_docs = []
    img_docs = []
    imgs = {}
    if markdown_path:
        with open(markdown_path / "metadata.txt", mode = 'r', encoding = 'utf-8') as f:
             metadata = f.read()
             metadata = metadata.split('\n')[1:]
             metadata = [i.split(' | ') for i in metadata]
             metadata = {
                  i[0]: {
                       "type": "text",
                       "title": i[0],
                       "authors": i[1],
                       "doc_type": i[2]
                  }
                  for i in metadata
             }
        
        for mkd in tqdm(markdown_path.glob("*.md"), dynamic_ncols=True, unit="doc", desc="Loading docs", leave=True):
            with open(mkd, mode = 'r', encoding = 'utf-8') as f:
                content = f.read()
                docs = splitter.split_text(content) if splitter else [Document(content)]
                for doc in docs: doc.metadata.update(metadata[mkd.stem])
                mkd_docs += docs
    
    if images_path:
        with open(images_path / "metadata.txt", mode = 'r', encoding= 'utf-8') as f:
             metadata = f.read()
             metadata = metadata.split('\n')[1:]
             metadata = [i.split(' | ') for i in metadata]
             metadata = {
                  i[0]: {
                       "type": "image",
                       "image": i[0],
                       "document": i[1],
                       "doc_type": i[2],
                       "page_no": i[3],
                       "caption": i[4]
                  }
                  for i in metadata
             }
        
        for img in tqdm(images_path.glob("*.png"), dynamic_ncols=True, unit="doc", desc="Loading docs", leave=True):
            with open(img, 'rb') as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                imgs[img.stem] = encoded_string
            doc = Document(
                 page_content = metadata[img.stem]['caption'],
                 metadata = metadata[img.stem]
            )
            img_docs.append(doc)
    
    return mkd_docs, img_docs, imgs


    def ingest_from_pdf(self, input_path: Path | list[Path] | None = None, ingest_from_main_source: bool = True, converter: DocumentConverter = di.default_converter):
        
        if input_path:
            if input_path.is_file():
                raise ValueError("Input path must be a directory or a list of file paths.")
        
        documents = []
        if ingest_from_main_source and self.source_path:
            documents.extend(di.read_documents(self.source_path, converter))
        
        if input_path:
            documents.extend(di.read_documents(input_path, converter))
        
        mkd_path = self.cwd / "markdown_files"
        img_path = self.cwd / "image_files"

        if not mkd_path.is_dir():
            mkd_path.mkdir()
        if not img_path.is_dir():
            img_path.mkdir()

        di.extract_markdown_images(documents, mkd_path, img_path)

        mkd_docs, img_docs, imgs = di.load_data(mkd_path, img_path, self.splitter)
        
        self.images.update(imgs)
        self.mkd_docs.extend(mkd_docs)
        self.img_docs.extend(img_docs)

        all_docs = mkd_docs + img_docs
        self.vectorstore.add_documents(all_docs)
        
        return True
    
    def ingest_from_mkd_imgs(self, mkd_dir: Path = None, imgs_dir: Path = None, splitter: TextSplitter | None = None):
        
        if not splitter: splitter = self.splitter
  
        mkd_docs, img_docs, imgs = di.load_data(mkd_dir, imgs_dir, splitter)
            
        self.images.update(imgs)
        self.mkd_docs.extend(mkd_docs)
        self.img_docs.extend(img_docs)
        
        all_docs = mkd_docs + img_docs
        self.vectorstore.add_documents(all_docs)
        
        return True