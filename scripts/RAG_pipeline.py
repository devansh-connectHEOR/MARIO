import scripts.data_ingestion as di
import scripts.utilities as ut

from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from docling.document_converter import DocumentConverter

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage


default_system_prompt = """
### ROLE
You are a high-stakes HEOR (Health Economics and Outcomes Research) Research Assistant. 
Your mission is to produce technically precise, audit-ready answers based strictly on the provided context.

### CORE PRINCIPLES
1. AUDIT-READY TRACEABILITY
   - Every factual claim MUST be followed by a citation in the format:
     [Document Name, Authors, Section X.Y]
   - If information cannot be found in the provided context, explicitly state:
     "Information not found in provided documentation."
   - Never hallucinate.
   - Never infer beyond what is supported by retrieved text.

2. FORMATTING
   - Ensure a professional tone.
   - Use clear structure (bullet points, headers) where appropriate.
   - Do not overstate findings.

### CONTEXT
{context}
"""

class RAG:
    def __init__(
            self, 
            pdf_docs_dir: Path | None = None, 
            working_dir_path: Path | None = None, 
            llm_model: str = "gpt-4o-mini", 
            embeddings_model: str = "text-embedding-3-small", 
            splitter: TextSplitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]), 
            system_prompt: str = default_system_prompt
    ):

        if pdf_docs_dir or working_dir_path:

            self.source_path = pdf_docs_dir if pdf_docs_dir else None
            self.cwd = working_dir_path if working_dir_path else Path.cwd() / "rag_working_dir"
            self.llm = ChatOpenAI(model=llm_model, temperature=0.0)
            self.images = {}
            self.mkd_docs = []
            self.img_docs = []
            self.embeddings = OpenAIEmbeddings(model=embeddings_model)
            self.splitter = splitter
            self.system_prompt = system_prompt

            if not self.cwd.is_dir():
                self.cwd.mkdir(parents=True)
                vectorstore_path = self.cwd / "vectorstore"
                vectorstore_path.mkdir(parents=False)
            else: 
                self.setup_from_working_dir()

            self.vectorstore = Chroma(
                collection_name="tsd_vector_store",
                embedding_function=self.embeddings,
                persist_directory=str(self.cwd / "vectorstore")
            )
            
            # 2. Setup standard LCEL Chain instead of an Agent
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{query}")
            ])
            self.chain = self.prompt_template | self.llm | StrOutputParser()

        else:
            raise ValueError("At least one of pdf_docs_path or working_dir_path must be provided.") 

    def setup_from_working_dir(self):
        mkd_path = self.cwd / "markdown_files"
        img_path = self.cwd / "image_files"

        if not mkd_path.is_dir() or not img_path.is_dir():
            raise ValueError("Working directory must contain 'markdown_files' and 'image_files' subdirectories.")

        mkd_docs, img_docs, imgs = di.load_data(mkd_path, img_path, self.splitter)
        
        self.images.update(imgs)
        self.mkd_docs.extend(mkd_docs)
        self.img_docs.extend(img_docs)

        return True
    
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
        if not splitter: 
            splitter = self.splitter
  
        mkd_docs, img_docs, imgs = di.load_data(mkd_dir, imgs_dir, splitter)
            
        self.images.update(imgs)
        self.mkd_docs.extend(mkd_docs)
        self.img_docs.extend(img_docs)
        
        all_docs = mkd_docs + img_docs
        self.vectorstore.add_documents(all_docs)
        
        return True
        
    class doc_type(BaseModel):
        doc_type: str = Field(description="Whether the document is a TSD (Technical Support Document) or a TA (Technical Assessment).")
    
    def create_multimodal_context(self, docs: list[Document]) -> tuple[str, list[str]]:
        """Separates text context and image base64 strings for multimodal formatting."""
        text_content = ["Context:\n"]
        image_b64_list = []
        
        text_docs = [d for d in docs if d.metadata.get("type") == "text"]
        img_docs = [d for d in docs if d.metadata.get("type") == "image"]
        
        for doc in text_docs:
            text_content.append(f"Document Title: {doc.metadata.get('title')}\nAuthors: {doc.metadata.get('authors')}\nDocument Type: {doc.metadata.get('doc_type')}\nContent: {doc.page_content}\n")

        for doc in img_docs:
            # Add metadata about the image to the text context
            text_content.append(f"Image Reference -> Document: {doc.metadata.get('document')}\nCaption: {doc.metadata.get('caption')}\nPage no: {doc.metadata.get('page_no')}\n")
            
            # Extract the actual base64 string
            img_b64 = self.images.get(doc.metadata.get('image'))
            if img_b64:
                image_b64_list.append(img_b64)
                
        return "\n\n".join(text_content), image_b64_list

    def retrieve_context_parts(self, query: str, k: int = 5) -> tuple[str, list[str]]:
        """Retrieves documents and returns separated text and image data."""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        doc_type_llm = self.llm.with_structured_output(self.doc_type)
        doc_type_output = doc_type_llm.invoke(query)
        target_doc_type = doc_type_output.doc_type if doc_type_output else None

        mkd_docs = self.mkd_docs 

        if target_doc_type:
            retriever.search_kwargs["filter"] = {"doc_type": target_doc_type}
            mkd_docs = [d for d in mkd_docs if d.metadata.get("doc_type") == target_doc_type]
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, BM25Retriever.from_documents(mkd_docs, search_kwargs={"k": k})] if mkd_docs else [retriever],
            weights=[0.5, 0.5] if mkd_docs else [1.0]
        )

        retrieved_docs = ensemble_retriever.invoke(query)
        return self.create_multimodal_context(retrieved_docs)

    def _build_messages(self, query: str, text_context: str, image_b64_list: list[str]) -> list:
        """Constructs the message payload required for vision models."""
        human_content = [
            {"type": "text", "text": f"{text_context}\n\nQuestion: {query}"}
        ]
        
        # Append images in the correct OpenAI format
        for img_b64 in image_b64_list:
            human_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"} # Adjust mime type if they are png
            })

        return [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=human_content)
        ]

    def answer(self, query: str, k: int = 5) -> str:
        """Answers queries using multimodal context."""
        text_context, image_b64_list = self.retrieve_context_parts(query, k)
        messages = self._build_messages(query, text_context, image_b64_list)
        
        response = self.llm.invoke(messages)
        return response.content