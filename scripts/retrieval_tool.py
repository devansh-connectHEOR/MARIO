from langchain_core.tools import BaseTool, ToolException
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_chroma import Chroma
from typing import Type, Any

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import asyncio

precise_retrieval_desctiption = """
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

class retrieval_tool(BaseTool):

    name: str 
    description: str 
    bm25_weight: float = Field(default=0.7, description="Weight for BM25 retriever in the ensemble.")
    k: int = Field(default=5, description="Number of top documents to retrieve.")
    documents: list[Document]
    images: dict[str, str]
    vectorstore: Any


    def create_context(self, docs: list[Document]) -> list[dict]:
        content = []
        for doc in docs:
            if doc.metadata.get("type") == "image":
                im_dict = [
                    {"type": "text",
                     "text": f"The following image is from document {doc.metadata.get('document')}, page {doc.metadata.get('page_no')}, titled '{doc.metadata.get('caption')}'."
                     },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{self.images.get(doc.metadata.get('image'))}",
                        "detail": "low"
                    }
                }]
                content.extend(im_dict)
            else:
                text_content = (
                    f"Document Title: {doc.metadata.get('title')}\n"
                    f"Authors: {doc.metadata.get('authors')}\n"
                    f"Document Type: {doc.metadata.get('doc_type')}\n"
                    f"Header 1: {doc.metadata.get('Header 1')}\n"
                    f"Header 2: {doc.metadata.get('Header 2')}\n"
                    f"Header 3: {doc.metadata.get('Header 3')}\n"
                    f"Content: {doc.page_content}"
                )
                tex_dict = {
                    "type": "text",
                    "text": text_content
                }
                content.append(tex_dict)
        return content


    class doc_type(BaseModel):
        doc_type: str = Field(description="Whether the documents required are TSD (Technical Support Document) or TA (Technical Assessment) or Both.")


    def _run(self, query: str) -> str:

        if not self.vectorstore:
            raise ToolException("Vectorestore not set for RetrievalTool.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        doc_type_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).with_structured_output(self.doc_type)
        doc_type = doc_type_llm.invoke(query)
        doc_type = doc_type.doc_type if doc_type.doc_type != "Both" else None
        documents = self.documents
        #print(f"Determined doc_type for retrieval: {doc_type}")
        if doc_type:
            retriever.search_kwargs["filter"] = {"doc_type": doc_type}
            documents = [d for d in documents if d.metadata.get("doc_type") == doc_type]
            #print(len(documents))
        bm25 = BM25Retriever.from_documents(documents, k = self.k)
        
        #print(len(bm25.invoke(query)), len(retriever.invoke(query)))
        
        ensemble_retriever = EnsembleRetriever(
            retrievers=[retriever, bm25],
            weights=[1.0 - self.bm25_weight, self.bm25_weight],

        )
        rrf_docs = ensemble_retriever.invoke(query)
        
        text_indices = [idx for idx, d in enumerate(rrf_docs) if d.metadata.get('type') == 'text']
        stop_idx = text_indices[self.k] if len(text_indices) > self.k else len(rrf_docs)
        retrieved_docs = rrf_docs[:stop_idx]
        
        #print(len(retrieved_docs))
        context = self.create_context(retrieved_docs)
        return context
        