import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
import re
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

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
    progress_bar = st.progress(0, text="Starting index build...")
    stage_text = st.empty()

    def on_progress(stage: str, current: int, total: int, detail: str) -> None:
        total_safe = max(total, 1)
        pct = int(min(100, max(0, (current / total_safe) * 100)))
        if stage == "prepare":
            progress_bar.progress(3, text="Preparing embedding model...")
            stage_text.info("Initializing embedding model (first run can take longer).")
            return
        if stage == "loading":
            progress_bar.progress(pct, text=f"Loading PDFs: {current}/{total_safe}")
            if detail:
                stage_text.info(f"Processing: {detail}")
            return
        if stage == "chunking":
            progress_bar.progress(92, text="Chunking loaded documents...")
            stage_text.info("Splitting document text into chunks.")
            return
        if stage == "embedding":
            progress_bar.progress(97, text="Building FAISS vector index...")
            stage_text.info("Generating embeddings and saving index.")
            return
        if stage == "done":
            progress_bar.progress(100, text="Index build complete.")
            stage_text.success("Indexing finished.")

    with st.spinner("Indexing PDFs..."):
        status = get_pipeline().load_and_process_documents(progress_callback=on_progress)
        if status and status.get("ok"):
            st.success(
                f"Index built successfully. PDFs: {status.get('pdf_count', 0)}, "
                f"loaded docs: {status.get('loaded_docs', 0)}, chunks: {status.get('chunks', 0)}"
            )
            failed = status.get("failed_files", []) or []
            if failed:
                st.warning(f"Some PDFs failed to parse ({len(failed)}): " + ", ".join(failed[:8]))
        else:
            msg = (status or {}).get("message", "Indexing failed.")
            st.error(msg)
            failed = (status or {}).get("failed_files", []) or []
            if failed:
                st.warning("Failed PDFs: " + ", ".join(failed[:12]))


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
        r"\b(flow\s*chart|flow\s*chat|process\s*flow|process\s*diagram|workflow\s*diagram"
        r"|flow\s*diagram|process\s*map|activity\s*diagram|swimlane|procedure\s*flow)\b",
        question, re.IGNORECASE,
    ))

def wants_table(question: str) -> bool:
    """Detect if the user is asking for a table."""
    return bool(re.search(
        r"\b(table|tabular|comparison\s*chart|matrix|list\s*all|summarize\s*in\s*table)\b",
        question, re.IGNORECASE,
    ))


def parse_markdown_table(table_md: str) -> pd.DataFrame:
    """Parse a markdown table string into a pandas DataFrame."""
    if not table_md or '|' not in table_md:
        return None
    
    lines = [line.strip() for line in table_md.split('\n') if line.strip()]
    if len(lines) < 3:  # Need at least header, separator, and one data row
        return None
    
    # Find the header row and separator
    header_idx = None
    separator_idx = None
    
    for i, line in enumerate(lines):
        # Skip lines like "--- Page 3 ---"
        if '---' in line and not line.startswith('|'):
            continue
            
        if line.startswith('|'):
            # Check if next line is separator line with dashes
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line.startswith('|') and '---' in next_line:
                    header_idx = i
                    separator_idx = i + 1
                    break
    
    if header_idx is None:
        return None
    
    # Parse header
    header_line = lines[header_idx].strip()
    headers = [h.strip() for h in header_line.split('|')[1:-1]]  # Remove empty strings from split
    
    if not headers:
        return None
    
    # Remove empty column headers and track which columns to keep
    valid_columns = [(i, h) for i, h in enumerate(headers) if h and h.strip()]
    
    if not valid_columns:
        return None
    
    # Get the header names for comparison (to skip duplicate headers in data)
    header_names_lowercase = [h.lower() for i, h in valid_columns]
    
    # Parse data rows
    data_rows = []
    for line in lines[separator_idx + 1:]:
        if line.startswith('|'):
            # Skip separator-like lines
            if '---' in line:
                continue
            # Skip page marker lines
            if 'page' in line.lower():
                continue
                
            row = [cell.strip() for cell in line.split('|')[1:-1]]
            if row:
                # Filter to keep only valid columns
                filtered_row = [row[i] if i < len(row) else '' for i, _ in valid_columns]
                
                # Skip if this row is a duplicate header row
                filtered_row_lower = [str(c).lower() for c in filtered_row]
                if filtered_row_lower == header_names_lowercase:
                    continue
                
                data_rows.append(filtered_row)
    
    if not data_rows:
        return None
    
    # Use only non-empty headers
    final_headers = [h for _, h in valid_columns]
    
    # Deduplicate column names by adding suffix if needed
    seen = {}
    final_headers_dedup = []
    for h in final_headers:
        if h in seen:
            seen[h] += 1
            final_headers_dedup.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            final_headers_dedup.append(h)
    
    df = pd.DataFrame(data_rows, columns=final_headers_dedup)
    return _normalize_table_alignment(df)


def _normalize_table_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common PDF-extraction misalignments before rendering."""
    if df is None or df.empty:
        return df

    work = df.copy()
    work = work.fillna("").astype(str)

    def is_empty_name(name: str) -> bool:
        n = str(name).strip().lower()
        return n == "" or n.startswith("unnamed")

    def non_empty_ratio(series: pd.Series) -> float:
        vals = series.astype(str).str.strip()
        return float((vals != "").sum()) / max(1, len(vals))

    # Pass 1: if an empty-header column carries data and the next named column is mostly empty,
    # shift values right into that named column.
    cols = list(work.columns)
    i = 0
    while i < len(cols) - 1:
        c = cols[i]
        n = cols[i + 1]
        if is_empty_name(c) and not is_empty_name(n):
            cur_ratio = non_empty_ratio(work[c])
            next_ratio = non_empty_ratio(work[n])
            if cur_ratio >= 0.45 and next_ratio <= 0.2:
                work[n] = work[n].where(work[n].astype(str).str.strip() != "", work[c])
                work = work.drop(columns=[c])
                cols = list(work.columns)
                continue
        i += 1

    # Pass 2: drop columns that are truly empty noise.
    drop_cols = []
    for c in work.columns:
        ratio = non_empty_ratio(work[c])
        if is_empty_name(c) and ratio < 0.15:
            drop_cols.append(c)
    if drop_cols:
        work = work.drop(columns=drop_cols)

    # Pass 3: use first row as sub-header hints for blank header columns when it looks like a sub-header row.
    if not work.empty:
        first = work.iloc[0].astype(str).str.strip().tolist()
        first_non_empty = sum(1 for v in first if v)
        if first_non_empty >= max(2, int(len(first) * 0.4)):
            renamed = False
            new_cols = []
            for idx, c in enumerate(work.columns):
                hint = first[idx] if idx < len(first) else ""
                if is_empty_name(c) and hint and len(hint.split()) <= 3:
                    new_cols.append(hint)
                    renamed = True
                else:
                    new_cols.append(c)
            if renamed:
                work.columns = new_cols
                # Remove the sub-header row if it mostly mirrors header words.
                header_tokens = [str(h).strip().lower() for h in work.columns]
                first_tokens = [str(v).strip().lower() for v in first]
                overlap = sum(1 for h, v in zip(header_tokens, first_tokens) if v and (v == h or v in h or h in v))
                if overlap >= max(2, int(len(work.columns) * 0.4)):
                    work = work.iloc[1:].reset_index(drop=True)

    # Final cleanup: drop rows that are fully empty after normalization.
    mask = work.apply(lambda r: any(str(x).strip() for x in r), axis=1)
    work = work[mask].reset_index(drop=True)

    # De-duplicate column names after adjustments.
    seen = {}
    dedup = []
    for c in work.columns:
        key = str(c).strip() or "col"
        if key in seen:
            seen[key] += 1
            dedup.append(f"{key}_{seen[key]}")
        else:
            seen[key] = 0
            dedup.append(key)
    work.columns = dedup

    return work


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
            with st.spinner("Cropping best flowchart image..."):
                result = pipeline.generate_flowchart(prompt)
                error = result.get("error", "")
                images = result.get("images", [])

                if error and not images:
                    st.warning(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})

                elif images:
                    for image_bytes in images:
                        st.image(image_bytes, use_container_width=True)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "",
                        "images": images,
                    })

                else:
                    fallback = (
                        "No flowchart image found. Try with exact SOP name and process name."
                    )
                    st.markdown(fallback)
                    st.session_state.messages.append({"role": "assistant", "content": fallback})

    # ─── TABLE REQUEST ───
    elif route == "table":
        with st.chat_message("assistant"):
            with st.spinner("📋 Extracting and analysing table with AI..."):
                result = pipeline.generate_table(prompt)
                multi_tables = result.get("multi_tables", [])
                table_md = result.get("table", "")
                sources = result.get("sources", [])
                error = result.get("error", "")
                llm_cleaned = result.get("llm_cleaned", False)
                llm_generated = result.get("llm_generated", False)

                if multi_tables:
                    rendered_any = False
                    message_blocks = []
                    for entry in multi_tables:
                        title = entry.get("title", "Table")
                        t_err = entry.get("error", "")
                        t_md = entry.get("table", "")
                        t_sources = entry.get("sources", [])

                        st.markdown(f"**📋 {title}:**")
                        if t_err and not t_md:
                            st.warning(t_err)
                            message_blocks.append(f"{title}: {t_err}")
                            continue

                        clean_md = re.sub(r'^---.*?---\n', '', t_md, flags=re.MULTILINE).strip()
                        df = parse_markdown_table(clean_md)
                        if df is not None and not df.empty:
                            st.dataframe(df, use_container_width=True)
                            rendered_any = True
                        elif clean_md:
                            st.markdown(clean_md)
                            rendered_any = True
                        else:
                            st.warning(f"No data extracted for {title}.")

                        if t_sources:
                            st.markdown("**Sources:** " + ", ".join(t_sources))

                        message_blocks.append(f"{title}\n\n{t_md or t_err}")

                    if not rendered_any:
                        fallback = "Could not generate the requested tables from the selected SOP."
                        st.markdown(fallback)
                        st.session_state.messages.append({"role": "assistant", "content": fallback})
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "\n\n".join(message_blocks),
                            "is_table": True,
                        })
                    # done handling multi-table request
                    table_md = ""

                if error:
                    st.warning(error)
                    st.session_state.messages.append({"role": "assistant", "content": error})
                elif table_md:
                    if llm_generated:
                        st.success("✨ Table generated by AI from SOP document context.")
                    elif llm_cleaned:
                        st.success("✨ Table extracted from PDF and refined by AI.")

                    # Strip the --- Page N --- header before parsing
                    clean_md = re.sub(r'^---.*?---\n', '', table_md, flags=re.MULTILINE).strip()
                    df = parse_markdown_table(clean_md)
                    if df is not None and not df.empty:
                        st.markdown("**📋 Table from SOP Documents:**")
                        st.dataframe(df, use_container_width=True)
                        if sources:
                            st.markdown("**Sources:** " + ", ".join(sources))
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"**📋 Table from SOP Documents:**\n\n{table_md}",
                            "is_table": True,
                        })
                    else:
                        st.markdown(clean_md)
                        if sources:
                            st.markdown("**Sources:** " + ", ".join(sources))
                        st.session_state.messages.append({"role": "assistant", "content": clean_md})
                else:
                    fallback = (
                        "Could not generate a table for this topic. "
                        "Try being more specific about what data you need."
                    )
                    st.markdown(fallback)
                    st.session_state.messages.append({"role": "assistant", "content": fallback})

    # ─── REGULAR QUESTION ───
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching all SOP documents..."):
                result = pipeline.answer_question(prompt)
                answer = result.get("answer", "Not found in the provided SOPs.")
                sources = result.get("sources", [])
                st.markdown(answer)
                if sources:
                    st.markdown("**Sources:** " + ", ".join(sources))

        st.session_state.messages.append({"role": "assistant", "content": answer})
