from langchain_core.tools import BaseTool, ToolException
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class RetrievalTool(BaseTool):
    name = "retrieval_tool"
    description = "Tool for retrieving information from the vector database. Use this tool to answer questions about the content of the documents in the knowledge base. Input should be a question or query related to the documents, and the output will be relevant information retrieved from the vector database."

    documents: list[Document]
    retriever: BaseRetriever

    def _run(self, query: str) -> str:
        if not self.retriever:
            raise ToolException("Retriever not set for RetrievalTool.")
        
        