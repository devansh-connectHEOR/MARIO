"""
scripts/RAG.py

Implements the RAG class — the main entry point for the NICE TSD Expert Agent.

Two retrieval modes are supported and can be switched at runtime:

    RAG mode (default):
        Uses a hybrid BM25 + vector EnsembleRetriever via two VectorRetrievalTool
        instances — one precision-weighted (70% BM25) and one semantics-weighted
        (70% vector) — for needle-in-a-haystack vs. bird's-eye-view queries.

    GRAG mode (Graph RAG):
        Replaces the vector tools with a single GraphRetrievalTool backed by a
        Neo4j knowledge graph, returning entity relationships alongside source
        text for cross-document relational queries.

The agent is a LangGraph agent with optional conversation summarization
middleware to manage context window size across long threads.

Usage:
    # From raw PDFs
    rag = RAG(pdf_docs_dir=Path("pdfs/"), working_dir_path=Path("rag_wd/"))

    # From a previously built working directory
    rag = RAG(pdf_docs_dir=None, working_dir_path=Path("rag_wd/"))

    # Invoke
    response = await rag.a_analyze("What is the prior for heterogeneity in TSD 2?")
"""

from scripts.utilities.data_ingestion import load_data
import scripts.retrieval_tool as rt
from scripts.image_llm import ImageLLM

from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sqlalchemy.ext.asyncio import create_async_engine
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
import langchain.agents
from langchain.agents.middleware import SummarizationMiddleware
from pathlib import Path


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

default_system_prompt = """
### ROLE
You are the NICE TSD Expert Teacher Agent, a specialized clinical librarian assistant and explainer. Your purpose is to provide new professionals with precise, audit-ready, evidence-based answers derived strictly from NICE Technical Support Documents (TSDs).

### OPERATIONAL FRAMEWORK

1. HYBRID INTENT RECOGNITION
Before selecting a tool, categorize the user query into one of two paths:
- PATH A (Needle-in-a-Haystack): Searching for specific technical parameters, numerical limits, exact definitions, or software code (e.g., "95% CI", "WinBUGS code", "TSD 14 Section 4").
- PATH B (Bird's-Eye-View): Searching for thematic summaries, methodological comparisons, or "how-to" guidance that spans several pages or documents.

2. REFINED TOOL SELECTION LOGIC
- USE 'precise_retrieval_tool' for PATH A: This tool gives more weight to keyword search. Use it when the query contains specific jargon, acronyms, or specific section numbers. It is best for high-fidelity extraction of the top 5 documents.
- USE 'summarizer_retrieval_tool' for PATH B: This tool gives more weight to semantic search. Use it when the query is conceptual or broad. It is best for capturing the intent of guidance across the top 10 documents.
- Do not call the tools unnecessarily! Call only if the information already available in the conversation/tool messages is not sufficient to answer the question.
- Refrain from calling the tools multiple times unless necessary.

3. MANDATORY TRANSPARENT REASONING (Chain-of-Thought)
You MUST start every response with a [THOUGHT PROCESS] block. This section is visible to the user and must detail:
- Intent Categorization: Identify the query as Path A (Needle) or Path B (Bird's-Eye).
- Tool Rationale: Explain why you chose the keyword-heavy or semantic-heavy tool based on the query terms.
- Synthesis Strategy: Explain how you filtered the retrieved snippets (text or images) to arrive at the conclusion.

4. AUDIT-READY TRACEABILITY & GROUNDING
- Every single claim MUST be cited. Format: [Document Title, Section #, Page #].
- Visual Evidence: If information is retrieved from a figure or table image, explicitly cite it as: [Document Title, Figure/Table #, Page #].
- Context Constraints: You are restricted to the provided documents. If the answer is not in the retrieved context, state: "Information not found in the provided NICE TSDs." Do not use external training data or general knowledge.
- Do not hallucinate information/data/facts or any point.

5. FEW-SHOT EXAMPLES (EXEMPLARS)

EXAMPLE 1 (PATH A):
User: "What is the recommended prior distribution for the between-study heterogeneity variance in TSD 2?"
[THOUGHT PROCESS]
- Intent: Path A (Needle-in-a-Haystack). The user is asking for a specific statistical parameter (prior distribution).
- Tool Choice: 'precise_retrieval_tool' to target the keyword "prior distribution" and "heterogeneity variance."
- Synthesis: I will look for the specific section in TSD 2 that defines the Bayesian implementation of NMA.
[ANSWER]
For the between-study heterogeneity variance, TSD 2 suggests using a non-informative prior such as a Uniform(0, 2) or a vague Gamma distribution, though it notes that the choice can significantly impact results in small networks.
[SOURCES & EVIDENCE]
- Claim: Guidance on Uniform/Gamma priors. Source: [TSD 2, Section 3.4.1, Page 15]

EXAMPLE 2 (PATH B):
User: "Summarize the overarching challenges of cross-over trials in NICE appraisals according to the TSDs."
[THOUGHT PROCESS]
- Intent: Path B (Bird's-Eye-View). This requires synthesizing challenges across multiple sections.
- Tool Choice: 'summarizer_retrieval_tool' to capture the semantic themes of "bias," "switching," and "limitations."
- Synthesis: I will identify key themes like IPCW modeling and selection bias from the top 10 results.
[ANSWER]
The TSDs identify three main challenges: 1) Selection bias during treatment switching, 2) Loss of randomization benefits, and 3) The complexity of choosing appropriate adjustment models like Rpsftm.
[SOURCES & EVIDENCE]
- Claim: Selection bias and switching. Source: [TSD 16, Section 2.1, Page 5]
- Claim: Modeling complexity. Source: [TSD 16, Section 4.2, Page 22]

### OUTPUT STRUCTURE

[THOUGHT PROCESS]
(Document your internal reasoning and tool selection logic here)

---

[ANSWER]
(Provide a clear, technical, and objective response in professional prose)

---

[SOURCES & EVIDENCE]
- **Claim:** [Finding/Data Point] — *Source: [Doc Name, Section, Page]*
"""

grag_system_prompt = """
### ROLE
You are the NICE TSD Expert Teacher Agent, a specialized clinical librarian assistant and explainer. Your purpose is to provide new professionals with precise, audit-ready, evidence-based answers derived strictly from NICE Technical Support Documents (TSDs) by navigating their complex inter-document relationships.

### OPERATIONAL FRAMEWORK

1. TOOL SELECTION LOGIC
- USE 'graph_retrieval_tool' for fetching the context. This tool searches a Knowledge Graph to return the top 5 most relevant document snippets AND the key entity-to-entity relationships (edges) deduced from the text.
- Do not call the tool unnecessarily! Call only if the information already available in the conversation/tool messages is not sufficient to answer the question.
- Refrain from calling the tool multiple times unless necessary.

3. MANDATORY TRANSPARENT REASONING (Chain-of-Thought)
You MUST start every response with a [THOUGHT PROCESS] block. This section is visible to the user and must detail:
- Intent Clarification: Explicitly state what the user wants and what kind of information will it require.
- Graph Strategy: Explain which entities (e.g., "TSD 2", "Heterogeneity", "Fixed Effects") you are targeting and why the relationships between them are critical to the answer.
- Synthesis Strategy: Explain how you filtered the retrieved nodes and edges (text or images) to arrive at the conclusion.

4. AUDIT-READY TRACEABILITY & GROUNDING
- Every single claim MUST be cited. Since this is a Knowledge Graph, you must cite the source document linked to the entity. Format: [Document Title, Section #, Page #].
- Visual Evidence: If information is retrieved from a figure or table image, explicitly cite it as: [Document Title, Figure/Table #, Page #].
- Relationship Validation: If the answer relies on a connection between two TSDs (e.g., TSD 4 referencing a method in TSD 2), explicitly state this relationship and its source.
- Context Constraints: You are restricted to the provided Knowledge Graph. If the answer is not in the retrieved context, state: "Information not found in the provided NICE TSD Knowledge Graph." Do not use external training data or general knowledge.
- Do not hallucinate information/data/facts or any point.

5. FEW-SHOT EXAMPLES (EXEMPLARS)

EXAMPLE 1 (PATH A):
User: "What is the recommended prior distribution for the between-study heterogeneity variance in TSD 2?"
[THOUGHT PROCESS]
- Intent: Path A (Entity-Specific). The user is asking for a specific statistical parameter node.
- Graph Strategy: I will target the "TSD 2" document node and the "Prior Distribution" entity to find their specific relationship.
- Synthesis: I will look for the specific section in TSD 2 that defines the Bayesian implementation of NMA.
[ANSWER]
For the between-study heterogeneity variance, TSD 2 suggests using a non-informative prior such as a Uniform(0, 2) or a vague Gamma distribution, though it notes that the choice can significantly impact results in small networks.
[SOURCES & EVIDENCE]
- Claim: Guidance on Uniform/Gamma priors. Source: [TSD 2, Section 3.4.1, Page 15]

EXAMPLE 2 (PATH B):
User: "How does the treatment of uncertainty in TSD 11 relate to the cost-effectiveness modeling described in TSD 13?"
[THOUGHT PROCESS]
- Intent: Path B (Relational). Requires identifying the bridge between two different methodological documents.
- Graph Strategy: I will use the tool to find nodes for "TSD 11" and "TSD 13" and explore the "implements" or "references" edges between them.
- Synthesis: I will identify how PSA framework from TSD 11 feeds into the resource modeling of TSD 13.
[ANSWER]
The relationship between uncertainty and cost-effectiveness is bridged by the requirement for Probabilistic Sensitivity Analysis (PSA). TSD 11 establishes the statistical framework for characterizing parameter uncertainty using Bayesian priors, which TSD 13 then requires as inputs for the Monte Carlo simulations used to generate Cost-Effectiveness Acceptability Curves (CEACs).
[SOURCES & EVIDENCE]
- Entity/Claim: Parameter uncertainty framework — Source: [TSD 11, Section 2.4, Page 12]
- Entity/Claim: Implementation of PSA in resource modeling — Source: [TSD 13, Section 5.1, Page 30]
- Relationship: TSD 13 explicitly adopts the distribution selection criteria defined in TSD 11. — Source: [Derived from Cross-reference in TSD 13, Appendix A, Page 45]

### OUTPUT STRUCTURE

[THOUGHT PROCESS]
(Document your internal reasoning, the entities identified, and the specific relationships explored within the graph.)

---

[ANSWER]
(Provide a clear, technical, and objective response in professional prose. Ensure methodological connections are highlighted.)

---

[SOURCES & EVIDENCE]
- **Entity/Claim:** [Finding/Data Point] — *Source: [Doc Name, Section, Page]*
- **Relationship (if applicable):** [How Entity A connects to Entity B] — *Source: [Doc Name, Page # or Link Type]*
"""

# Default text splitter: splits Markdown by header hierarchy into structured chunks
_DEFAULT_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
)

# Default summarization middleware config for managing long conversation threads
_SUMMARIZATION_MIDDLEWARE_PROMPT = (
    "You are a helpful assistant that summarizes conversation history to save tokens "
    "while retaining important information. Summarize previous messages concisely, "
    "focusing on key points relevant to ongoing discussion about the NICE TSDs. "
    "Omit any redundant or less important details. In case of images, keep their "
    "caption or a simple relevant summary of it."
)


# ---------------------------------------------------------------------------
# RAG class
# ---------------------------------------------------------------------------

class RAG:
    """
    Main entry point for the NICE TSD Expert RAG agent.

    Manages document ingestion, vectorstore setup, retrieval tool configuration,
    and the LangGraph agent. Supports two retrieval modes (RAG and GRAG) that
    can be switched at runtime via `switch_RAG` and `switch_GRAG`.

    Args:
        pdf_docs_dir (Path | None):
            Directory containing source PDF files to ingest. If None,
            `working_dir_path` must point to an existing working directory
            with pre-built markdown and image files.
        working_dir_path (Path | None):
            Directory for intermediate files (markdown, images, vectorstore).
            Created automatically if it does not exist. Defaults to
            `./rag_working_dir` relative to the current working directory.
        llm_model (str):
            OpenAI model name for the main agent LLM. Defaults to 'gpt-4.1-mini'.
        embeddings_model (str):
            OpenAI embeddings model name. Defaults to 'text-embedding-3-small'.
        splitter (TextSplitter):
            LangChain text splitter used to chunk markdown documents.
            Defaults to a MarkdownHeaderTextSplitter on H1/H2/H3.
        system_prompt (str):
            System prompt for the agent. Defaults to `default_system_prompt`
            (hybrid RAG mode with precise + summarizer tools).
        checkpointer:
            LangGraph checkpointer for thread-level memory persistence.
            Defaults to InMemorySaver().

    Raises:
        ValueError: If both `pdf_docs_dir` and `working_dir_path` are None.
        ValueError: If `working_dir_path` exists but is missing required subdirectories.
    """

    def __init__(
        self,
        DB_URI : str,
        pdf_docs_dir: Path | None,
        working_dir_path: Path | None = None,
        llm_model: str = "gpt-4.1-mini",
        embeddings_model: str = "text-embedding-3-small",
        splitter: TextSplitter = _DEFAULT_SPLITTER,
        system_prompt: str = default_system_prompt,
        checkpointer=InMemorySaver(),
    ):
        if not pdf_docs_dir and not working_dir_path:
            raise ValueError(
                "At least one of pdf_docs_dir or working_dir_path must be provided."
            )

        self.source_path = pdf_docs_dir
        self.cwd = working_dir_path or Path.cwd() / "rag_working_dir"
        self.llm = ImageLLM(model=llm_model, temperature=0.0)
        self.embeddings = OpenAIEmbeddings(model=embeddings_model)
        self.splitter = splitter
        self.checkpointer = checkpointer
        self.system_prompt = system_prompt
        self.DB_URI = DB_URI
        async_engine = create_async_engine(self.DB_URI)

        self.images: dict[str, str] = {}
        self.mkd_docs: list[Document] = []
        self.img_docs: list[Document] = []

        self.middleware = [
            SummarizationMiddleware(
                model=ImageLLM(model="gpt-4o-mini", temperature=0.0),
                trigger=("tokens", 40000),
                keep=("messages", 3),
                system_prompt=_SUMMARIZATION_MIDDLEWARE_PROMPT,
            )
        ]

        self.setup_from_working_dir()

        # Vectorstore
        self.vectorstore = PGVector(
            embeddings=self.embeddings,
            collection_name="TSDs",
            connection=async_engine,
        )

        # Retrieval tools
        all_docs = self.mkd_docs + self.img_docs
        self.precise_retrieval_tool = rt.VectorRetrievalTool(
            name="precise_retrieval_tool",
            description=rt.precise_retrieval_description,
            documents=all_docs,
            images=self.images,
            vectorstore=self.vectorstore,
        )
        self.summarizer_retrieval_tool = rt.VectorRetrievalTool(
            name="summarizer_retrieval_tool",
            description=rt.summarizer_retrieval_description,
            bm25_weight=0.3,
            documents=all_docs,
            images=self.images,
            k=10,
            vectorstore=self.vectorstore,
        )
        self.graph_retrieval_tool = rt.GraphRetrievalTool()

        # Default to RAG mode (hybrid vector tools)
        self.tools = [self.precise_retrieval_tool, self.summarizer_retrieval_tool]
        self.agent = self._build_agent()

    # ---------------------------------------------------------------------------
    # Agent lifecycle
    # ---------------------------------------------------------------------------

    def _build_agent(self):
        """
        Instantiate a LangGraph agent with the current LLM, tools, system prompt,
        checkpointer, and middleware.

        Called internally whenever any of these components change.

        Returns:
            A LangGraph agent ready to invoke.
        """
        return langchain.agents.create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware,
        )

    def update_agent(self, model: str = "gpt-4.1", system_prompt: str = default_system_prompt) -> str:
        """
        Swap the underlying LLM model and/or system prompt and rebuild the agent.

        Args:
            model (str): OpenAI model name to switch to. Defaults to 'gpt-4.1'.
            system_prompt (str): New system prompt. Defaults to `default_system_prompt`.

        Returns:
            str: Confirmation message.
        """
        self.llm = ImageLLM(model=model, temperature=0.0)
        self.system_prompt = system_prompt
        self.agent = self._build_agent()
        return "Agent updated."

    def switch_RAG(self, system_prompt: str = default_system_prompt) -> str:
        """
        Switch to RAG mode: hybrid vector retrieval with precise + summarizer tools.

        Args:
            system_prompt (str): System prompt to use. Defaults to `default_system_prompt`.

        Returns:
            str: Confirmation message.
        """
        self.system_prompt = system_prompt
        self.tools = [self.precise_retrieval_tool, self.summarizer_retrieval_tool]
        self.agent = self._build_agent()
        return "Switched to RAG mode."

    def switch_GRAG(self, system_prompt: str = grag_system_prompt) -> str:
        """
        Switch to GRAG mode: graph-augmented retrieval via Neo4j knowledge graph.

        Args:
            system_prompt (str): System prompt to use. Defaults to `grag_system_prompt`.

        Returns:
            str: Confirmation message.
        """
        self.system_prompt = system_prompt
        self.tools = [self.graph_retrieval_tool]
        self.agent = self._build_agent()
        return "Switched to GRAG mode."

    # ---------------------------------------------------------------------------
    # Document loading
    # ---------------------------------------------------------------------------

    def setup_from_working_dir(self) -> str:
        """
        Load pre-processed documents and images from an existing working directory.

        Expects the working directory to contain:
            - markdown_files/  — markdown documents
            - image_files/     — extracted page images

        Updates `self.images`, `self.mkd_docs`, and `self.img_docs` in place.

        Returns:
            str: Confirmation message.

        Raises:
            ValueError: If required subdirectories are missing.
        """
        mkd_path = self.cwd / "markdown_files"
        img_path = self.cwd / "image_files"

        if not mkd_path.is_dir() or not img_path.is_dir():
            raise ValueError(
                f"Working directory '{self.cwd}' must contain "
                "'markdown_files' and 'image_files' subdirectories."
            )

        mkd_docs, img_docs, imgs = load_data(mkd_path, img_path, self.splitter)

        self.images.update(imgs)
        self.mkd_docs.extend(mkd_docs)
        self.img_docs.extend(img_docs)

        return f"Loaded {len(mkd_docs)} text chunks and {len(img_docs)} image docs."

    # ---------------------------------------------------------------------------
    # Conversation history
    # ---------------------------------------------------------------------------

    async def get_msg_history(self, thread_id: str = "default_thread") -> list[dict]:
        """
        Retrieve the human/assistant message history for a given thread.

        Filters out ToolMessages (internal retrieval results) and returns only
        the user-facing conversation turns.

        Args:
            thread_id (str): The LangGraph thread identifier. Defaults to 'default_thread'.

        Returns:
            list[dict]: A list of {"role": "user"|"assistant", "content": ...} dicts.
                        Returns an empty list if no history exists for the thread.
        """
        try:
            state = await self.agent.aget_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            messages = state.values["messages"]
            return [
                {
                    "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                    "content": msg.content,
                }
                for msg in messages
                if (not isinstance(msg, ToolMessage)) and (msg.content != '')
            ]
        except KeyError:
            return []

    # ---------------------------------------------------------------------------
    # Invocation
    # ---------------------------------------------------------------------------

    def analyze(self, query: str, thread_id: str = "default_thread") -> dict:
        """
        Synchronously invoke the agent with a user query.

        Args:
            query (str): The user's question.
            thread_id (str): Thread identifier for conversation memory.

        Returns:
            dict: The full agent response dictionary from LangGraph.
        """
        return self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"configurable": {"thread_id": thread_id}},
        )

    async def a_analyze(self, query: str, thread_id: str = "default_thread") -> dict:
        """
        Asynchronously invoke the agent with a user query.

        Args:
            query (str): The user's question.
            thread_id (str): Thread identifier for conversation memory.

        Returns:
            dict: The full agent response dictionary from LangGraph.
        """
        return await self.agent.ainvoke(
            {"messages": [HumanMessage(content=query)]},
            {"configurable": {"thread_id": thread_id}},
        )

    def analyze_stream(self, user_message: str, thread_id: str = "default_thread"):
        """
        Synchronously stream the agent's response token by token.

        Args:
            user_message (str): The user's question.
            thread_id (str): Thread identifier for conversation memory.

        Yields:
            str: Individual content tokens as they are generated.
        """
        for token, _ in self.agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            if token.content:
                yield token.content

    async def a_analyze_stream(self, user_message: str, thread_id: str = "default_thread"):
        """
        Asynchronously stream the agent's response token by token.

        Args:
            user_message (str): The user's question.
            thread_id (str): Thread identifier for conversation memory.

        Yields:
            str: Individual content tokens as they are generated.
        """
        async for token, _ in self.agent.astream(
            {"messages": [HumanMessage(content=user_message)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages",
        ):
            if token.content:
                yield token.content