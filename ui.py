import time
import uuid
from shiny import App, ui, reactive, render
from scripts.RAG import RAG
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. Purely Synchronous Agent Logic
# No asyncio needed. This is a standard Python generator.
# ---------------------------------------------------------------------------
def run_sync_langchain_agent(prompt: str, config: dict):
    # In a real setup, you would use your synchronous agent_executor.stream() here
    yield "Let me plan out how to answer that...\n\n"
    time.sleep(1)  # Standard blocking sleep
    
    yield f"🔍 *Retrieving context (Top K: {config['top_k']})*...\n\n"
    time.sleep(1.5)
    
    yield f"⚙️ *Synthesizing response using {config['model']}*...\n\n"
    time.sleep(1)
    
    yield "Here is the final synthesized answer using purely synchronous logic."

# ---------------------------------------------------------------------------
# 2. UI Layout
# ---------------------------------------------------------------------------
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("Pipeline Configuration"),
        ui.input_select("model_choice", "LLM Provider", {"gpt-4o": "GPT-4o", "gemini-1.5-pro": "Gemini 1.5 Pro"}),
        ui.input_numeric("top_k", "Top-K Chunks", value=5, min=1, max=20),
        
        ui.hr(),
        
        ui.h4("System Status"),
        ui.output_ui("status_indicator"),
        
        ui.hr(),
        ui.input_action_button("clear_chat", "Clear Chat", class_="btn-danger"),
        
        width=320
    ),
    ui.card(
        ui.card_header("Agentic RAG Interface"),
        ui.chat_ui("rag_chat"),
        full_screen=True
    ),
    title="Agentic RAG Dashboard",
    fillable=True
)

# ---------------------------------------------------------------------------
# 3. Server Logic
# ---------------------------------------------------------------------------
def server(input, output, session):
    chat = ui.Chat(id="rag_chat")
    current_status = reactive.Value("Idle 🟢")
    thread_id = reactive.Value(str(uuid.uuid4()))

    cwd = Path(r"C:\Users\Public\Documents\MARIO\rag_working_dir")
    rag = RAG(None, cwd)

    @render.ui
    def status_indicator():
        is_idle = "Idle" in current_status()
        bg_color = "#28a745" if is_idle else "#ffc107"
        text_color = "white" if is_idle else "black"
        
        return ui.HTML(
            f"<div style='padding: 10px; border-radius: 6px; background-color: {bg_color}; "
            f"color: {text_color}; font-weight: bold; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>"
            f"{current_status()}"
            f"</div>"
        )

    # The callback MUST remain async so it can trigger the UI WebSocket, 
    # but it calls your completely synchronous backend code.
    @chat.on_user_submit
    async def handle_submit():
        user_message = chat.user_input()
        current_status.set("Agent Processing ⏳")
        
        # Call the standard, non-async generator
        stream = rag.analyze_stream(user_message, thread_id)
        
        # Await the UI update to pipe the sync stream to the frontend
        await chat.append_message_stream(stream)
        
        current_status.set("Idle 🟢")

    @reactive.effect
    @reactive.event(input.clear_chat)
    async def reset_session():
        await chat.clear_messages()
        thread_id.set(str(uuid.uuid4()))
        current_status.set("Idle 🟢")

app = App(app_ui, server)