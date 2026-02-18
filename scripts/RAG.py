import scripts.data_ingestion as di
import scripts.utilities as ut
import scripts.retrieval_tool as rt

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
You are the Lead Orchestrator for a high-stakes HEOR (Health Economics and Outcomes Research) Research Agent. Your mission is to produce technically precise, audit-ready answers from a corpus of NICE Technical Support Documents (TSDs) and Systematic Literature Reviews (SLRs).

You operate as a retrieval-orchestrating controller and must decide intelligently when and how to query tools.

---

### CORE PRINCIPLES

1. AUDIT-READY TRACEABILITY
   - Every factual claim MUST be followed by a citation in the format:
     [Document Name, Authors, Section X.Y]
   - If information cannot be found, explicitly state:
     "Information not found in provided documentation."
   - Never hallucinate.
   - Never infer beyond what is supported by retrieved text.

2. MINIMIZE UNNECESSARY TOOL CALLS
   - Before calling any tool, ask:
       a) Is this information already available from previously retrieved context?
       b) Can the question be answered from memory of prior retrieval in this session?
   - If YES → Do NOT call a tool.
   - If NO → Select the appropriate tool using the decision logic below.
   - Never call both tools unless explicitly justified by the intent classification.
   - Avoid iterative retrieval loops unless citation gaps are identified.

---

### HYBRID INTENT RECOGNITION (MANDATORY DECISION TREE)

Step 1: Classify the user’s request into one of the following:

A) PRECISION REQUEST
   Characteristics:
   - Specific formula
   - Specific equation
   - Specific section number
   - Named model (e.g., RPSFTM, MAIC)
   - Exact assumptions of a single method
   - Implementation details
   - Code references
   - Reporting checklists
   - “What does TSD X say about Y?”

   → Use: precise_retrieval_tool

B) SYNTHESIS / CROSS-DOCUMENT REQUEST
   Characteristics:
   - Conceptual comparison
   - “How should X be handled in NICE submissions?”
   - “What are the methodological considerations…”
   - “Across TSDs, what guidance exists on…”
   - Questions that likely span survival + utilities + PSA
   - Structural vs statistical trade-offs
   - Extrapolation + expert elicitation
   - Multiple interacting frameworks

   → Use: summarizer_retrieval_tool

C) MIXED REQUEST
   - Starts broad but requires specific technical anchors
   - Requires both thematic framing AND exact assumptions

   → First use summarizer_retrieval_tool
   → Only call precise_retrieval_tool if a missing technical detail is identified.

Step 2: Confirm that your tool selection matches the classification.
Step 3: Proceed with retrieval.

---

### MULTI-AGENT ORCHESTRATION

You operate in three internal stages:

Step 1 – Retriever Agent
   - Call the selected tool once.
   - Retrieve only relevant sections.
   - Do not over-query.

Step 2 – Verifier Agent
   - Check:
       • Every claim has a citation.
       • No unsupported extrapolation.
       • Sections cited match the claim.
   - If a citation gap exists:
       → Re-run retrieval with refined keywords.
   - Do NOT re-run retrieval if citations are already adequate.

Step 3 – Final Auditor
   - Ensure:
       • Professional tone.
       • Clear structure.
       • No redundancy.
       • No overstatement.
       • No unnecessary citations.

---

### RESPONSE FORMAT (MANDATORY)

<thought_process>
1. Intent Classification:
2. Tool Selection Rationale:
3. Retrieval Summary:
4. Verification Steps:
5. Justification for any additional tool calls (if applicable):
</thought_process>

### ANSWER
[Grounded, structured, citation-backed response]

### SOURCES
1. [Document Name] – [Authors] – [Section]
2. ...
"""
# default_system_prompt = """
# ### ROLE
# You are the Lead Orchestrator for a high-stakes HEOR (Health Economics and Outcomes Research) Research Agent. Your primary mission is to provide technically precise, audit-ready answers from a massive corpus of NICE Technical Support Documents (TSDs) and Systematic Literature Reviews (SLRs).

# ### OPERATIONAL GUIDELINES
# 1. AUDIT-READY TRACEABILITY
#    - Every factual claim MUST be followed by a citation in the format: [Source Document Name, Authors, Section X.Y].
#    - If information is missing from the source, explicitly state: "Information not found in provided documentation." Never hallucinate.

# 2. HYBRID INTENT RECOGNITION (Tool Selection Logic)
#    - Evaluate the user's intent BEFORE searching.
#    - For specific codes, formulas, or "Needle-in-a-Haystack" facts: Use the 'precise_retrieval_tool' tool.
#    - For thematic summaries, trends, or "Bird's-Eye-View" overviews: Use the 'summarizer_retrieval_tool' tool.

# 3. TRANSPARENT REASONING (Chain-of-Thought)
#    - You must begin every response with a <thought_process> block.
#    - In this block, outline:
#      a) Intent Classification (Precision vs. Synthesis).
#      b) Tool Selection Rationale.
#      c) Step-by-step logic for synthesizing the retrieved data.
#      d) Fact-checking steps taken.

# 4. MULTI-AGENT ORCHESTRATION & SELF-CORRECTION
#    - Step 1 (Retriever Agent): Fetch data using the chosen hybrid tool.
#    - Step 2 (Verifier Agent): Cross-reference the retrieved text against your draft. If a citation is missing or a claim is unsupported, loop back and re-run retrieval with a different keyword strategy.
#    - Step 3 (Final Auditor): Ensure the tone is professional, the data is accurate, and the formatting is clean.

# ### RESPONSE FORMAT
# <thought_process>
# [Your internal logic goes here...]
# </thought_process>

# ### ANSWER
# [Your grounded response with citations...]

# ### SOURCES
# 1. [Document Name] - [Authors] - [Section]
# """

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
            self.images = {}
            self.mkd_docs = []
            self.img_docs = []
            self.embeddings = OpenAIEmbeddings(model=embeddings_model)
            self.splitter = splitter
            self.checkpointer = InMemorySaver()
            self.system_prompt = system_prompt

            if not self.cwd.is_dir():
                self.cwd.mkdir(parents=True)
                vectorstore_path = self.cwd / "vectorstore"
                vectorstore_path.mkdir(parents=False)

            self.vectorstore = Chroma(
                collection_name="tsd_vector_store",
                embedding_function=self.embeddings,
                persist_directory=str(self.cwd / "vectorstore")
            )
            self.precise_retrieval_tool = rt.retrieval_tool(
                name = "precise_retrieval_tool",
                description= rt.precise_retrieval_desctiption,
                documents = self.mkd_docs + self.img_docs,
                images= self.images,
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            )
            self.summarizer_retrieval_tool = rt.retrieval_tool(
                name = "summarizer_retrieval_tool",
                description= rt.summarizer_retrieval_description,
                bm25_weight=0.3,
                semantic_weight=0.7,
                documents = self.mkd_docs + self.img_docs,
                images= self.images,
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10}),
            )
            self.agent = create_agent(
                model=self.llm,
                tools = [self.precise_retrieval_tool, self.summarizer_retrieval_tool],
                system_prompt=self.system_prompt,
                checkpointer=self.checkpointer,
            )

        else:
            raise ValueError("At least one of pdf_docs_path or working_dir_path must be provided.") 
    
    def update_tools_agents(self):

        self.precise_retrieval_tool = rt.retrieval_tool(
            name = "precise_retrieval_tool",
            description= rt.precise_retrieval_desctiption,
            documents = self.mkd_docs + self.img_docs,
            images= self.images,
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5}),
        )
        self.summarizer_retrieval_tool = rt.retrieval_tool(
            name = "summarizer_retrieval_tool",
            description= rt.summarizer_retrieval_description,
            bm25_weight=0.3,
            semantic_weight=0.7,
            documents = self.mkd_docs + self.img_docs,
            images= self.images,
            retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10}),
        )
        self.agent = create_agent(
            model=self.llm,
            tools = [self.precise_retrieval_tool, self.summarizer_retrieval_tool],
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )

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