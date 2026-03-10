import scripts.data_ingestion as di
import scripts.retrieval_tool as rt
from scripts.image_llm import image_llm

from langchain_core.documents import Document
from langchain_text_splitters.base import TextSplitter
from docling.document_converter import DocumentConverter
import pydantic
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
import langchain.agents
from langgraph.checkpoint.memory import InMemorySaver
from langchain.messages import HumanMessage, ToolMessage
from langchain.agents.middleware import SummarizationMiddleware


default_system_prompt = """
### ROLE
You are the NICE TSD Expert Teacher Agent, a specialized clinical librarian assistant adn explainer. Your purpose is to provide new professionals with precise, audit-ready, evidence-based answers derived strictly from NICE Technical Support Documents (TSDs). 

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
            llm_model: str = "gpt-4.1", 
            embeddings_model: str = "text-embedding-3-small", 
            splitter: TextSplitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]), 
            system_prompt: str = default_system_prompt,
            checkpointer = InMemorySaver()
    ):

        if pdf_docs_dir or working_dir_path:

            self.source_path = pdf_docs_dir if pdf_docs_dir else None
            self.cwd = working_dir_path if working_dir_path else Path.cwd() / "rag_working_dir"
            self.llm = image_llm(model=llm_model, temperature=0.0)
            self.images = {}
            self.mkd_docs = []
            self.img_docs = []
            self.embeddings = OpenAIEmbeddings(model=embeddings_model)
            self.splitter = splitter
            self.checkpointer = checkpointer
            self.system_prompt = system_prompt
            self.middleware = [
                SummarizationMiddleware(
                    model=image_llm(model = "gpt-4o-mini", temperature=0.0),
                    trigger=("tokens", 40000),
                    keep=("messages", 3),
                    system_prompt="You are a helpful assistant that summarizes conversation history to save tokens while retaining important information. Summarize previous messages concisely, focusing on key points relevant to ongoing discussion about the NICE TSDs. Omit any redundant or less important details. In case of images, keep their caption or a simple relevant summary of it."
                )
            ]

            if not self.cwd.is_dir():
                self.cwd.mkdir(parents=True)
                vectorstore_path = self.cwd / "vectorstore"
                vectorstore_path.mkdir(parents=False)
            
            else: self.setup_from_working_dir()

            self.vectorstore = Chroma(
                collection_name="tsd_vector_store",
                embedding_function=self.embeddings,
                persist_directory=str(self.cwd / "vectorstore2")
            )
            self.precise_retrieval_tool = rt.vector_retrieval_tool(
                name = "precise_retrieval_tool",
                description= rt.precise_retrieval_desctiption,
                documents = self.mkd_docs + self.img_docs,
                images= self.images,
                vectorstore= self.vectorstore,
            )
            self.summarizer_retrieval_tool = rt.vector_retrieval_tool(
                name = "summarizer_retrieval_tool",
                description= rt.summarizer_retrieval_description,
                bm25_weight=0.3,
                documents = self.mkd_docs + self.img_docs,
                images= self.images,
                k=10,
                vectorstore= self.vectorstore,
            )
            self.graph_retrieval_tool = rt.graph_retrieval_tool()
            self.agent = langchain.agents.create_agent(
                model=self.llm,
                tools = [self.precise_retrieval_tool, self.summarizer_retrieval_tool],
                system_prompt=self.system_prompt,
                checkpointer=self.checkpointer,
                middleware=self.middleware
            )

        else:
            raise ValueError("At least one of pdf_docs_path or working_dir_path must be provided.") 
    
    def update_tools_agents(self):

        self.precise_retrieval_tool = rt.vector_retrieval_tool(
            name = "precise_retrieval_tool",
            description= rt.precise_retrieval_desctiption,
            documents = self.mkd_docs + self.img_docs,
            images= self.images,
            vectorstore= self.vectorstore,
        )
        self.summarizer_retrieval_tool = rt.vector_retrieval_tool(
            name = "summarizer_retrieval_tool",
            description= rt.summarizer_retrieval_description,
            bm25_weight=0.3,
            documents = self.mkd_docs + self.img_docs,
            images= self.images,
            k=10,
            vectorstore= self.vectorstore,
        )
        self.agent = langchain.agents.create_agent(
            model=self.llm,
            tools = [self.precise_retrieval_tool, self.summarizer_retrieval_tool],
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware
        )

        return True

    def switch_RAG(self, system_prompt = default_system_prompt):
        self.system_prompt = system_prompt
        self.agent = langchain.agents.create_agent(
            model=self.llm,
            tools = [self.precise_retrieval_tool, self.summarizer_retrieval_tool],
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware
        )

        return "RAG setup complete"
    
    def switch_GRAG(self, system_prompt = grag_system_prompt):
        self.system_prompt = grag_system_prompt
        self.agent = langchain.agents.create_agent(
            model=self.llm,
            tools = [self.graph_retrieval_tool],
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            middleware=self.middleware
        )

        return "GRAG setup complete"       

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
        
        if not splitter: splitter = self.splitter
  
        mkd_docs, img_docs, imgs = di.load_data(mkd_dir, imgs_dir, splitter)
            
        self.images.update(imgs)
        self.mkd_docs.extend(mkd_docs)
        self.img_docs.extend(img_docs)
        
        all_docs = mkd_docs + img_docs
        self.vectorstore.add_documents(all_docs)
        
        return True
        
    def get_msg_history(self, thread_id='default_thread'):
        try:
            msgs = self.agent.get_state(config = {"configurable": {"thread_id":thread_id}}).values['messages']
            msgs = [i for i in msgs if not isinstance(i, ToolMessage)]
            msgs = [
            {"role":"user" if isinstance(i, HumanMessage) else "assistant", "content": i.content} for i in msgs
            ]
        except KeyError:
            msgs = []
        return msgs

    def analyze(self, query:str, thread_id='default_thread'):
        response = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            {"configurable": {"thread_id": thread_id}}
        )
        return response
    
    def analyze_stream(self, user_message, thread_id='default_thread'):
        """
        Analyzes the user's message using the agent in a streaming manner.
        Args:
            user_message: The message from the user to be analyzed.
            thread_id: The thread ID for maintaining conversation context.
        """
        for token, metadata in self.agent.stream(
            {"messages": [HumanMessage(content=user_message)]},
            {"configurable": {"thread_id": thread_id}},
            stream_mode="messages"
        ):
            if token.content:
                yield token.content