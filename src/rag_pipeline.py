import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda

from src.flowchart_extractor import FlowchartExtractor
from src.table_extractor import TableExtractor
from src.text_pipeline import TextPipeline

# Load .env from project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)


class SOPRagPipeline:
    """Production-ready multi-modal RAG orchestrator.

    Strictly separated pipelines:
    1) Text pipeline    -> Q&A from PDF text chunks
    2) Table pipeline   -> deterministic table extraction (no LLM generation)
    3) Image pipeline   -> flowchart/image extraction from PDF pages
    """

    def __init__(
        self,
        pdf_dir: str,
        vector_db_path: str = "faiss_index",
        llm_provider: str = "",
        groq_model: str = "",
        groq_base_url: str = "",
    ):
        self.pdf_dir = pdf_dir
        self.vector_db_path = vector_db_path

        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "")
        self.groq_base_url = groq_base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

        self.llm = None

        images_dir = str(Path(__file__).resolve().parent.parent / "images")

        self.text_pipeline = TextPipeline(
            pdf_dir=self.pdf_dir,
            vector_db_path=self.vector_db_path,
            llm=None,
        )
        self.table_pipeline = TableExtractor(pdf_dir=self.pdf_dir, llm=None)
        self.flowchart_pipeline = FlowchartExtractor(pdf_dir=self.pdf_dir, images_dir=images_dir, llm=None)

        # Keep compatibility with existing app checks.
        self.vector_store = self.text_pipeline.vector_store

    def _setup_llm(self):
        provider = (self.llm_provider or "").lower().strip()
        if provider == "local":
            return self._setup_local_llm()
        if provider == "groq" or self.groq_api_key:
            return self._setup_groq_llm()
        return self._setup_local_llm()

    def _setup_local_llm(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        model_id = "MBZUAI/LaMini-T5-738M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=384, do_sample=False)
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return RunnableLambda(generate_text)

    def _setup_groq_llm(self):
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq API usage.")
        if not self.groq_model:
            raise ValueError("GROQ_MODEL is required for Groq API usage.")

        endpoint = f"{self.groq_base_url.rstrip('/')}/chat/completions"

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                prompt = prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

            payload = {
                "model": self.groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 1024,
            }
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }

            response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            if response.status_code >= 400:
                raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")
            data = response.json()
            return data["choices"][0]["message"]["content"]

        return RunnableLambda(generate_text)

    # ---------- Shared app-facing API ----------
    def load_and_process_documents(
        self,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ):
        status = self.text_pipeline.load_and_process_documents(progress_callback=progress_callback)
        self.vector_store = self.text_pipeline.vector_store
        return status

    def load_vector_store(self):
        self.text_pipeline.load_vector_store()
        self.vector_store = self.text_pipeline.vector_store

    def route_query(self, question: str) -> str:
        q = (question or "").lower()
        if any(k in q for k in ["flowchart", "flow chart", "flow chat", "process flow", "workflow", "diagram"]):
            return "flowchart"
        if any(k in q for k in [
            "table",
            "tabular",
            "raci",
            "sipoc",
            "matrix",
            "change history",
            "revision history",
            "change log",
            "revision log",
            "version history",
        ]):
            return "table"
        return "text"

    def answer_question(self, question: str) -> Dict[str, Any]:
        if self.llm is None:
            self.llm = self._setup_llm()
            self._propagate_llm()
        return self.text_pipeline.answer_question(question)

    def _propagate_llm(self):
        """Share the initialized LLM with flowchart and table pipelines."""
        self.text_pipeline.llm = self.llm
        self.flowchart_pipeline.llm = self.llm

    def generate_table(self, question: str) -> Dict[str, Any]:
        # Keep table extraction deterministic from PDFs.
        matched_pdf = self._match_pdf_file(question)
        q = (question or "").lower()

        wants_raci = "raci" in q
        wants_sipoc = "sipoc" in q
        wants_change_history = any(k in q for k in ["change history", "revision history", "change log"])
        wants_all_tables = any(k in q for k in ["all tables", "all the tables", "3 tables", "three tables"])

        requested_types = []
        if wants_raci:
            requested_types.append("raci")
        if wants_sipoc:
            requested_types.append("sipoc")
        if wants_change_history:
            requested_types.append("change_history")

        if wants_all_tables and len(requested_types) < 2:
            requested_types = ["raci", "sipoc", "change_history"]

        if len(requested_types) >= 2:
            titles = {
                "raci": "RACI Table",
                "sipoc": "SIPOC Table",
                "change_history": "Change History Table",
            }
            out_tables = []
            for table_type in requested_types:
                t_result = self.table_pipeline.extract_table(
                    question,
                    matched_pdf=matched_pdf,
                    forced_table_type=table_type,
                )
                out_tables.append(
                    {
                        "table_type": table_type,
                        "title": titles.get(table_type, table_type.replace("_", " ").title()),
                        "table": t_result.get("table", ""),
                        "sources": t_result.get("sources", []),
                        "error": t_result.get("error", ""),
                    }
                )

            return {
                "multi_tables": out_tables,
                "error": "",
            }

        return self.table_pipeline.extract_table(question, matched_pdf=matched_pdf)

    def generate_flowchart(self, question: str) -> Dict[str, Any]:
        # Ensure LLM is ready before flowchart extraction
        if self.llm is None:
            self.llm = self._setup_llm()
            self._propagate_llm()
        matched_pdf = self._match_pdf_file(question)
        out = self.flowchart_pipeline.extract_flowcharts(question, matched_pdf=matched_pdf, max_images=2)
        return {
            "mermaid": "",
            "sources": out.get("sources", []),
            "error": out.get("error", ""),
            "images": out.get("images", []),
        }

    def _normalize_text(self, text: str) -> str:
        import re

        lowered = (text or "").lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
        return " ".join(cleaned.split())

    def _match_pdf_file(self, question: str):
        q = self._normalize_text(question)
        if not q:
            return None

        root = Path(self.pdf_dir)
        if not root.exists():
            return None

        best = None
        best_len = 0
        for pdf in root.glob("*.pdf"):
            stem = self._normalize_text(pdf.stem)
            if stem and stem in q and len(stem) > best_len:
                best = pdf.name
                best_len = len(stem)

            words = set(stem.split()) - {"sop", "ut", "of", "and", "the", "in", "for", "a", "an", "to"}
            overlap = words & set(q.split())
            if len(words) > 0 and len(overlap) >= max(2, int(len(words) * 0.5)) and len(overlap) > best_len:
                best = pdf.name
                best_len = len(overlap)
        return best
