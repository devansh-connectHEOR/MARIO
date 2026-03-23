"""
scripts/retrieval_tool.py

Defines LangChain tools used by the RAG agent to retrieve context from
document corpora. Two retrieval strategies are provided:

VectorRetrievalTool:
    Hybrid search combining BM25 (keyword) and vector (semantic) retrieval via
    an EnsembleRetriever. BM25 indices are pre-built per doc_type at init time
    for fast filtering. A lightweight LLM call classifies each query to
    select the appropriate index. Returns a mixed content list of text and
    image blocks ready for a multimodal LLM.

GraphRetrievalTool:
    Semantic search over a Neo4j knowledge graph. Augments retrieved text
    chunks with extracted graph relationships (e.g. entity co-mentions),
    giving the agent structural cross-document context. Lazily initialised
    on first use.

Tool descriptions (precise_retrieval_description, summarizer_retrieval_description)
are defined as module-level constants and passed in at instantiation, allowing
the same class to serve different retrieval personas with different BM25/semantic
weight splits.
"""

from langchain_core.tools import BaseTool, ToolException
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field, PrivateAttr
from typing import Any


# --- Tool descriptions ---
# Passed as the `description` field at instantiation to control how the agent
# decides which tool to call and when.

precise_retrieval_description = """
A high-precision technical retrieval tool optimized for HEOR and NICE Technical Support Documents (TSD).
Use this tool when the user requires specific methodological details, statistical formulas,
NICE DSU guidance, or precise technical nomenclature.

This tool uses a 70/30 Hybrid Search (70% Keyword, 30% Semantic), making it superior for:
1. Identifying specific document references (e.g., 'TSD 14', 'RPSFTM').
2. Finding exact statistical procedures within dense technical manuals.
3. Conducting systematic literature review (SLR) screening based on technical inclusion/exclusion criteria.

Do NOT use this tool for general summaries or vague conceptual questions.
Use it for 'needle-in-a-haystack' queries where exact terminology and precision are critical.
"""

summarizer_retrieval_description = """
A broad-spectrum synthesis tool optimized for identifying themes, patterns, and 'bird's-eye-view'
summaries across multiple documents.

This tool uses a 30/70 Hybrid Search (30% Keyword, 70% Semantic), making it superior for:
1. Identifying common trends across different NICE TSDs or HEOR studies.
2. Summarizing the general 'consensus' on a methodological topic.
3. Exploratory research where the user does not know the specific technical terminology or TSD numbers.

Use this tool when the query is conceptual (e.g., 'What is the general approach to...')
rather than specific (e.g., 'What is the formula in TSD 14?').
It is engineered to prioritize 'theme-matching' over 'exact-word-matching.'
"""

# --- Neo4j Cypher query for graph-augmented retrieval ---
# Finds the source Document chunk for each matched node, then collects any
# graph relationships (excluding MENTIONS edges) to surface cross-document
# structural context alongside the raw text.
graph_retrieval_query = """
MATCH (node)<-[:MENTIONS]-(doc:Document)
OPTIONAL MATCH (node)-[rel]-(neighbor)
WHERE type(rel) <> 'MENTIONS'
WITH doc, node, score, collect(node.id + ' ' + type(rel) + ' ' + neighbor.id) AS relationships
RETURN
    "Original Source Text:\n" + doc.text +
    "\n\nExtracted Relationships:\n" +
    reduce(s="", r IN relationships | s + r + '\n') AS text,
    score,
    {} AS metadata
"""

class DocTypeClassification(BaseModel):
    """Structured output schema for doc_type classification."""
    doc_type: str = Field(
        description="Whether the documents required are 'TSD' (Technical Support Document), 'TA' (Technical Assessment), or Both."
    )

class VectorRetrievalTool(BaseTool):
    """
    A hybrid BM25 + vector retrieval tool for the RAG agent.

    At initialisation, builds BM25 indices over the full document set and
    for each known doc_type ('TSD', 'TA'). At query time, a small LLM call
    classifies the query to select the matching BM25 index and apply a
    vector store filter, then an EnsembleRetriever combines both signals
    via Reciprocal Rank Fusion (RRF).

    Returned context is a list of content blocks (text dicts and image_url
    dicts) ready to be inserted directly into a multimodal LLM message.
    """

    name: str
    description: str
    bm25_weight: float = Field(
        default=0.7,
        description="Weight assigned to the BM25 retriever in the ensemble (0–1). "
                    "The vector retriever receives 1 - bm25_weight."
    )
    k: int = Field(default=5, description="Number of top text documents to return.")
    documents: list[Document]
    images: dict[str, str]  # Maps image ID → base64-encoded PNG string
    vectorstore: Any

    _bm25_cache: dict = PrivateAttr(default_factory=dict)
    _doc_type_llm: Any = PrivateAttr()

    def __init__(self, **kwargs):
        """
        Initialise the tool, pre-building BM25 indices for each known doc_type.

        Indices are cached in `_bm25_cache` keyed by doc_type string, with
        None as the key for the full-corpus (unfiltered) index.
        """
        super().__init__(**kwargs)

        # Full-corpus BM25 index (used when query spans both doc types)
        self._bm25_cache[None] = BM25Retriever.from_documents(self.documents, k=self.k)

        # Per-doc_type BM25 indices for filtered retrieval
        doc_types = ["TSD", "TA"]  # Note: TA documents not yet ingested
        for dt in doc_types:
            filtered = [d for d in self.documents if d.metadata.get("doc_type") == dt]
            if not filtered:
                print(f"Warning: No documents found for doc_type='{dt}', skipping BM25 index.")
                continue
            self._bm25_cache[dt] = BM25Retriever.from_documents(filtered, k=self.k)

        # Small LLM used to classify queries as TSD, TA, or Both
        self._doc_type_llm = ChatOpenAI(
            model="gpt-4.1-mini", temperature=0
        ).bind_tools([DocTypeClassification], tool_choice="any")


    def create_context(self, docs: list[Document]) -> list[dict]:
        """
        Convert a list of retrieved Documents into a multimodal content block list.

        Text documents are formatted with their metadata headers for context.
        Image documents are formatted as image_url blocks with a descriptive
        caption prefix, using base64-encoded PNG data from `self.images`.

        Args:
            docs (list[Document]): Retrieved documents, potentially mixed text and image types.

        Returns:
            list[dict]: A list of content blocks (text or image_url dicts) ready
                        for insertion into a multimodal LLM message.
        """
        content = []

        for doc in docs:
            if doc.metadata.get("type") == "image":
                content.extend([
                    {
                        "type": "text",
                        "text": (
                            f"The following image is from document "
                            f"'{doc.metadata.get('document')}', "
                            f"page {doc.metadata.get('page_no')}, "
                            f"titled '{doc.metadata.get('caption')}'."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.images.get(doc.metadata.get('image'))}",
                            "detail": "low",
                        },
                    },
                ])
            else:
                content.append({
                    "type": "text",
                    "text": (
                        f"Document Title: {doc.metadata.get('title')}\n"
                        f"Authors: {doc.metadata.get('authors')}\n"
                        f"Document Type: {doc.metadata.get('doc_type')}\n"
                        f"Header 1: {doc.metadata.get('Header 1')}\n"
                        f"Header 2: {doc.metadata.get('Header 2')}\n"
                        f"Header 3: {doc.metadata.get('Header 3')}\n"
                        f"Content: {doc.page_content}"
                    ),
                })

        return content

    def _run(self, query: str) -> list[dict]:
        """
        Synchronous retrieval: classify query, build ensemble, retrieve and format context.

        Args:
            query (str): The user's retrieval query.

        Returns:
            list[dict]: Multimodal content blocks for the agent's context window.

        Raises:
            ToolException: If no vectorstore has been set on the tool.
        """
        if not self.vectorstore:
            raise ToolException("Vectorstore not set for VectorRetrievalTool.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        response = self._doc_type_llm.invoke(query)
 
        if response.tool_calls:
            extracted_args = response.tool_calls[0].get("args", {})
            doc_type_val = extracted_args.get("doc_type", "Both")
        else:
            doc_type_val = "Both"

        doc_type = doc_type_val if doc_type_val != "Both" else None

        if doc_type:
            retriever.search_kwargs["filter"] = {"doc_type": {"$eq": doc_type}}

        bm25 = self._bm25_cache.get(doc_type)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25],
            weights=[1.0 - self.bm25_weight, self.bm25_weight],
        )

        rrf_docs = ensemble_retriever.invoke(query)
        retrieved_docs = self._trim_to_k_text_docs(rrf_docs)

        return self.create_context(retrieved_docs)

    async def _arun(self, query: str) -> list[dict]:
        """
        Async retrieval: classify query, build ensemble, retrieve and format context.

        Args:
            query (str): The user's retrieval query.

        Returns:
            list[dict]: Multimodal content blocks for the agent's context window.
        """
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        response = await self._doc_type_llm.ainvoke(query)
        
        if response.tool_calls:
            extracted_args = response.tool_calls[0].get("args", {})
            doc_type_val = extracted_args.get("doc_type", "Both")
        else:
            doc_type_val = "Both"

        doc_type = doc_type_val if doc_type_val != "Both" else None

        if doc_type:
            retriever.search_kwargs["filter"] = {"doc_type": doc_type}

        bm25 = self._bm25_cache.get(doc_type)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25],
            weights=[1.0 - self.bm25_weight, self.bm25_weight],
        )

        rrf_docs = await ensemble_retriever.ainvoke(query)
        retrieved_docs = self._trim_to_k_text_docs(rrf_docs)

        return self.create_context(retrieved_docs)

    def _trim_to_k_text_docs(self, docs: list[Document]) -> list[Document]:
        """
        Trim the RRF-ranked document list to contain at most `k` text documents.

        Images are always included; only text documents count toward the limit.
        Stops at the first index after the k-th text document is encountered.

        Args:
            docs (list[Document]): RRF-ranked documents (mixed text and image).

        Returns:
            list[Document]: Trimmed document list with at most k text entries.
        """
        text_indices = [
            idx for idx, d in enumerate(docs) if d.metadata.get("type") == "text"
        ]
        stop_idx = text_indices[self.k] if len(text_indices) > self.k else len(docs)
        return docs[:stop_idx]


class GraphRetrievalTool(BaseTool):
    """
    A Neo4j graph-augmented retrieval tool for cross-document relationship context.

    Uses a Neo4jVector store with a custom Cypher retrieval query to return
    source text chunks enriched with extracted entity relationships from the
    knowledge graph. Lazily initialises the Neo4j connection and embeddings
    on first use.
    """

    name: str = "graph_retrieval_tool"
    description: str = (
        "Retrieves context and cross-document entity relationships from the "
        "knowledge graph. Use when structural or relational context between "
        "concepts and documents is needed."
    )

    _embeddings: Any = PrivateAttr(default=None)
    _graph_retriever: Any = PrivateAttr(default=None)

    @property
    def graph_retriever(self):
        """
        Lazily initialise and return the Neo4j vector retriever.

        The Neo4j connection and embeddings model are created on first access
        to avoid startup costs and allow the tool to be instantiated before
        the graph is available.

        Returns:
            VectorStoreRetriever: A retriever backed by the Neo4j vector index.
        """
        if self._graph_retriever is None:
            self._embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            self._graph_retriever = Neo4jVector.from_existing_index(
                embedding=self._embeddings,
                index_name="vector",
                retrieval_query=graph_retrieval_query,
            ).as_retriever(search_kwargs={"k": 5})
        return self._graph_retriever

    def _run(self, query: str) -> list[Document]:
        """
        Synchronous graph retrieval.

        Args:
            query (str): The user's retrieval query.

        Returns:
            list[Document]: Retrieved documents with graph relationship context.
        """
        return self.graph_retriever.invoke(query)

    async def _arun(self, query: str) -> list[Document]:
        """
        Async graph retrieval.

        Args:
            query (str): The user's retrieval query.

        Returns:
            list[Document]: Retrieved documents with graph relationship context.
        """
        return await self.graph_retriever.ainvoke(query)