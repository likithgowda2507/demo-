import os
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv

# Load .env from the project root
_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

import streamlit as st
import streamlit.components.v1 as components

from src.rag_pipeline import SOPRagPipeline


st.set_page_config(
    page_title="Quality SOP Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Space Grotesk', 'Trebuchet MS', 'Segoe UI', sans-serif;
    }
    .stApp {
        background: #ffffff;
        color: #000000;
    }
    .block-container {
        padding-top: 2rem;
    }
    .chat-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .chat-subtitle {
        color: #333333;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        background: #ffffff;
        border: 1px solid #dddddd;
    }
    .stChatMessage * {
        color: #000000 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

base_dir = Path(__file__).resolve().parent
pdf_dir = str(base_dir / "pdfs")
vector_db = str(base_dir / "faiss_index")
provider = os.getenv("LLM_PROVIDER", "")
model_name = os.getenv("GROQ_MODEL", "")
base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

col_left, _ = st.columns([1, 1])
with col_left:
    build_index = st.button("Build / Refresh Index")
st.caption("Tip: run index after adding new PDFs.")


def get_pipeline() -> SOPRagPipeline:
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = SOPRagPipeline(
            pdf_dir=pdf_dir,
            vector_db_path=vector_db,
            llm_provider=provider,
            groq_model=model_name,
            groq_base_url=base_url,
        )
        # Auto-load existing index on first start
        st.session_state.pipeline.load_vector_store()
        st.session_state.vector_loaded = st.session_state.pipeline.vector_store is not None
    return st.session_state.pipeline

if build_index:
    with st.spinner("Indexing PDFs..."):
        get_pipeline().load_and_process_documents()
        st.success("Index built successfully.")


st.markdown("<div class='chat-title'>Quality SOP Chat Assistant</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='chat-subtitle'>Ask questions, request flowcharts, or ask for tables from your SOP PDFs.</div>",
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Ask me anything about your SOP documents. I can also generate **flowcharts** and **tables** — just ask!"}
    ]

# ──────────────────────────── Intent Detection ────────────────────────────

def wants_flowchart(question: str) -> bool:
    """Detect if the user is asking for a flowchart/process diagram."""
    return bool(re.search(
        r"\b(flow\s*chart|flow\s*chat|process\s*flow|process\s*diagram|workflow\s*diagram|flow\s*diagram)\b",
        question, re.IGNORECASE,
    ))

def wants_table(question: str) -> bool:
    """Detect if the user is asking for a table."""
    return bool(re.search(
        r"\b(table|tabular|comparison\s*chart|matrix|list\s*all|summarize\s*in\s*table)\b",
        question, re.IGNORECASE,
    ))


# ──────────────────────────── Mermaid Renderer ────────────────────────────

def render_mermaid(mermaid_code: str, height: int = 600) -> None:
    """Render a Mermaid diagram in Streamlit using an HTML component."""
    if not mermaid_code:
        return

    # Escape for safe embedding in HTML
    escaped = mermaid_code.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <style>
            body {{
                background: transparent;
                display: flex;
                justify-content: center;
                padding: 20px;
                margin: 0;
            }}
            .mermaid {{
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 24px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            .mermaid svg {{
                max-width: 100%;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
{escaped}
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'dark',
                themeVariables: {{
                    primaryColor: '#1a3a5c',
                    primaryTextColor: '#e6eefb',
                    primaryBorderColor: '#3fd0ff',
                    lineColor: '#3fd0ff',
                    secondaryColor: '#18c29c',
                    tertiaryColor: '#1a2445',
                    fontFamily: 'Space Grotesk, sans-serif',
                }}
            }});
        </script>
    </body>
    </html>
    """
    components.html(html, height=height, scrolling=True)


# ──────────────────────────── Render Chat History ────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Re-render Mermaid diagrams from chat history
        if msg.get("mermaid"):
            render_mermaid(msg["mermaid"])
        if msg.get("images"):
            for image_bytes in msg["images"]:
                st.image(image_bytes)


# ──────────────────────────── Chat Input ────────────────────────────

prompt = st.chat_input("Ask a question about your SOPs")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    pipeline = get_pipeline()
    route = pipeline.route_query(prompt)

    # ─── FLOWCHART REQUEST ───
    if route == "flowchart":
        with st.chat_message("assistant"):
            with st.spinner("Generating flowchart from SOP documents..."):
                result = pipeline.generate_flowchart(prompt)
                mermaid_code = result.get("mermaid", "")
                sources = result.get("sources", [])
                error = result.get("error", "")
                images = result.get("images", [])

                if error:
                    st.warning(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})
                elif images:
                    st.markdown("**Flowchart image extracted from PDF:**")
                    for image_bytes in images:
                        st.image(image_bytes)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(sources))
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Flowchart image extracted from PDF.",
                        "images": images,
                    })
                elif mermaid_code:
                    st.markdown("**📊 Flowchart generated from SOP documents:**")
                    render_mermaid(mermaid_code)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(sources))

                    msg_content = f"📊 Flowchart generated.\n\n**Sources:** {', '.join(sources)}"
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": msg_content,
                        "mermaid": mermaid_code,
                    })
                else:
                    fallback = "Could not generate a flowchart for this topic. Try asking about a specific SOP procedure."
                    st.markdown(fallback)
                    st.session_state.messages.append({"role": "assistant", "content": fallback})

    # ─── TABLE REQUEST ───
    elif route == "table":
        with st.chat_message("assistant"):
            with st.spinner("Generating table from SOP documents..."):
                result = pipeline.generate_table(prompt)
                table_md = result.get("table", "")
                sources = result.get("sources", [])
                error = result.get("error", "")

                if error:
                    st.warning(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})
                elif table_md:
                    st.markdown(table_md)
                    if sources:
                        st.markdown("**Sources:** " + ", ".join(sources))
                    st.session_state.messages.append({"role": "assistant", "content": table_md})
                else:
                    fallback = "Could not generate a table for this topic. Try being more specific about what data you need."
                    st.markdown(fallback)
                    st.session_state.messages.append({"role": "assistant", "content": fallback})

    # ─── REGULAR QUESTION ───
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching all SOP documents..."):
                result = pipeline.answer_question(prompt)
                answer = result.get("answer", "Not found in the provided SOPs.")
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
