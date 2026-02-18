from langchain_core.tools import BaseTool, ToolException
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from langchain_core.documents import Document

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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
    semantic_weight: float = Field(default=0.3, description="Weight for semantic retriever in the ensemble.")
    documents: list[Document]
    images: dict[str, str]
    retriever: BaseRetriever

    def create_context(self, docs: list[Document]) -> str:
        content = ["Context:\n"]
        text_docs = [d for d in docs if d.metadata.get("type") == "text"]
        img_docs = [d for d in docs if d.metadata.get("type") == "image"]
        for doc in text_docs:
            content.append(f"type: text\nDocument Title: {doc.metadata.get('title')}\nAuthors: {doc.metadata.get('authors')}\nDocument Type: {doc.metadata.get('doc_type')}\nContent: {doc.page_content}\n")

        for doc in img_docs:
            content.append(f"type: image\nDocument Title: {doc.metadata.get('document')}\nDocument Type: {doc.metadata.get('doc_type')}\nImage Description: {doc.metadata.get('caption')}\nPage no: {doc.metadata.get('page_no')}\nImage: {self.images.get(doc.metadata.get('image'))}")
        return "\n\n".join(content)


    class doc_type(BaseModel):
        doc_type: str = Field(description="Whether the documents required are TSD (Technical Support Document) or TA (Technical Assessment) or Both.")


    def _run(self, query: str) -> str:
        if not self.retriever:
            raise ToolException("Retriever not set for RetrievalTool.")

        doc_type_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3).with_structured_output(self.doc_type)
        doc_type = doc_type_llm.invoke(query)
        doc_type = doc_type.doc_type if doc_type != "Both" else None
        if doc_type:
            self.retriever.search_kwargs["filter"] = {"doc_type": doc_type}
            documents = [d for d in self.documents if d.metadata.get("doc_type") == doc_type]
        bm25 = BM25Retriever.from_documents(self.documents, search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[self.retriever, bm25],
            weights=[0.3, 0.7]
        )
        retrieved_docs = ensemble_retriever.invoke(query)
        context = self.create_context(retrieved_docs)
        return context
        