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
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage


default_system_prompt = """
### ROLE
You are an expert Health Economist and Outcomes Research (HEOR) Analyst specializing in systematic literature reviews, health technology assessments (HTA), and economic modeling. Your objective is to provide high-precision technical answers grounded strictly in the provided context.

### OPERATIONAL GUIDELINES (STRICT COMPLIANCE REQUIRED)
1. **Zero Hallucination Policy:** Answer ONLY using the information explicitly stated in the provided context. If the answer is not in the context, state: "The provided context does not contain sufficient information to answer this question." Do not use external knowledge or general healthcare training.
2. **Contextual Fidelity:** Preserve the exact wording for technical terms (e.g., "Incremental Cost-Effectiveness Ratio", "Quality-Adjusted Life Years", "Markov Model state transitions").
3. **Citations:** Every factual claim, data point, or methodology choice MUST be followed by a citation in [Title/Author/Doc_Type] format corresponding to the metadata in the provided context.
4. **Data Integrity:** When tables are provided, extract data exactly as presented. Do not infer trends or perform calculations unless specifically requested and supported by raw numbers in the text.

### RESPONSE STRUCTURE
Your response must follow this exact sequence:

#### 1. <INTERNAL_REASONING> (Chain of Thought)
- Break down the user's query into technical requirements.
- Identify specific sections of the context that contain relevant evidence.
- Plan the logical flow of the answer (e.g., Methodology -> Results -> Limitations).
- Identify any potential gaps in the context that prevent a full answer.
*(Note: This section is for your internal logical validation to ensure accuracy.)*

#### 2. <HEOR_ANALYSIS>
- Provide the final answer in professional, technical language appropriate for a HTA submission.
- Use bullet points for multiple findings.
- Ensure every statement is cited (e.g., "[TSD 3, Dias et al., Technical Document]").

#### 3. <EVIDENCE_SUMMARY_TABLE>
- If quantitative data is present, summarize it in a Markdown table for quick reference.
"""

class RAG:
    def __init__(
            self, 
            pdf_docs_dir: Path | None, 
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
            self.embeddings = OpenAIEmbeddings(model=embeddings_model)
            self.splitter = splitter
            self.checkpointer = InMemorySaver()
            self.images = {}
            self.mkd_docs = []
            self.img_docs = []
            self.system_prompt = system_prompt

            if not self.cwd.is_dir():
                self.cwd.mkdir(parents=True)
                vectorstore_path = self.cwd / "vectorstore"
                vectorstore_path.mkdir(parents=False)

            self.vectorstore = Chroma(
                collection_name="knowledge_base",
                embedding_function=self.embeddings,
                persist_directory=str(self.cwd / "vectorstore")
            )

            self.agent = create_agent(
                model=self.llm,
                system_prompt=self.system_prompt,
                checkpointer=self.checkpointer,
            )

        else:
            raise ValueError("At least one of pdf_docs_path or working_dir_path must be provided.") 

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
    
    def ingest_from_mkd_imgs(self, mkd_dir: Path, imgs_dir: Path, splitter: TextSplitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])):
        
        mkd_docs, img_docs, imgs = di.load_data(mkd_dir, imgs_dir, splitter)
        
        self.images.update(imgs)
        self.mkd_docs.extend(mkd_docs)
        self.img_docs.extend(img_docs)

        all_docs = mkd_docs + img_docs
        self.vectorstore.add_documents(all_docs)
        
        return True
        

    class doc_type(BaseModel):
        doc_type: str = Field(description="Whether the document is a TSD (Technical Support Document) or a TA (Technical Assessment).")
    
    def create_context(self, docs: list[Document]) -> str:
        content = ["Context:\n"]
        text_docs = [d for d in docs if d.metadata.get("type") == "text"]
        img_docs = [d for d in docs if d.metadata.get("type") == "image"]
        for doc in text_docs:
            content.append(f"type: text\nDocument Title: {doc.metadata.get('title')}\nAuthors: {doc.metadata.get('authors')}\nDocument Type: {doc.metadata.get('doc_type')}\nContent: {doc.page_content}\n")

        for doc in img_docs:
            content.append(f"type: image\nDocument Title: {doc.metadata.get('document')}\nDocument Type: {doc.metadata.get('doc_type')}\nImage Description: {doc.metadata.get('caption')}\nPage no: {doc.metadata.get('page_no')}\nImage: {self.images.get(doc.metadata.get('image'))}")
        return "\n\n".join(content)
    
    def query_with_context(self, query: str, k: int = 5) -> str:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        doc_type_llm = llm.with_structured_output(self.doc_type)
        doc_type = doc_type_llm.invoke(query)
        doc_type = doc_type.doc_type if doc_type else None

        mkd_docs = self.mkd_docs    #For keyword search using BM25

        if doc_type:
            retriever.search_kwargs["filter"] = {"doc_type": doc_type}
            mkd_docs = [d for d in mkd_docs if d.metadata.get("doc_type") == doc_type]
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, BM25Retriever.from_documents(mkd_docs, search_kwargs={"k": k})] if mkd_docs else [retriever],
            weights=[0.5, 0.5] if mkd_docs else [1.0]
        )

        retrieved_docs = ensemble_retriever.invoke(query)
        context = self.create_context(retrieved_docs)
        augmented_query = f"{context}\n\nQuestion: {query}"

        return augmented_query
    
    def answer_with_context(self, query: str, k: int = 5, thread_id='default_thread'):
        augmented_query = self.query_with_context(query, k)
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=augmented_query)]},
            {"configurable": {"thread_id": thread_id}}
        )
        return response

    def answer(self, query:str, thread_id='default_thread'):
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"configurable": {"thread_id": thread_id}}
        )
        return response