import os
import re
import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


_VECTOR_STORE_CACHE: Dict[str, FAISS] = {}


class TextPipeline:
    """Text-only RAG pipeline. Uses PDF text chunks and never answers without context."""

    def __init__(
        self,
        pdf_dir: str,
        vector_db_path: str,
        llm: Any,
    ):
        self.pdf_dir = pdf_dir
        self.vector_db_path = vector_db_path
        self.llm = llm
        self.embeddings: Optional[HuggingFaceEmbeddings] = None
        self.vector_store: Optional[FAISS] = None
        self._can_use_unstructured = self._check_unstructured_runtime()

    def _get_embeddings(self) -> HuggingFaceEmbeddings:
        if self.embeddings is None:
            self.embeddings = self._init_embeddings()
        return self.embeddings

    def _init_embeddings(self) -> HuggingFaceEmbeddings:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        local_kwargs = {"model_kwargs": {"local_files_only": True}}

        # First prefer local cache to avoid network/client-state issues during app startup.
        try:
            return HuggingFaceEmbeddings(model_name=model_name, **local_kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            client_err = any(k in msg for k in ["client has been closed", "cannot send a request"])
            auth_err = any(k in msg for k in ["401", "token", "unauthorized", "repositorynotfound", "expired"])

            # If local cache is missing, try online load.
            try:
                return HuggingFaceEmbeddings(model_name=model_name)
            except Exception as online_exc:
                online_msg = str(online_exc).lower()
                online_client_err = any(k in online_msg for k in ["client has been closed", "cannot send a request"])
                online_auth_err = any(
                    k in online_msg for k in ["401", "token", "unauthorized", "repositorynotfound", "expired"]
                )

                # Retry without auth tokens for public model when token state is bad.
                if online_auth_err or auth_err:
                    for key in [
                        "HF_TOKEN",
                        "HUGGINGFACEHUB_API_TOKEN",
                        "HF_API_TOKEN",
                        "HUGGING_FACE_HUB_TOKEN",
                    ]:
                        os.environ.pop(key, None)

                    try:
                        return HuggingFaceEmbeddings(model_name=model_name)
                    except Exception:
                        pass

                # Last fallback: local-only again, useful when HTTP client lifecycle is broken.
                if client_err or online_client_err:
                    return HuggingFaceEmbeddings(model_name=model_name, **local_kwargs)

                raise

            if not (auth_err or client_err):
                raise

            # Final retry without auth tokens for public model.
            for key in [
                "HF_TOKEN",
                "HUGGINGFACEHUB_API_TOKEN",
                "HF_API_TOKEN",
                "HUGGING_FACE_HUB_TOKEN",
            ]:
                os.environ.pop(key, None)

            return HuggingFaceEmbeddings(model_name=model_name)

    def _check_unstructured_runtime(self) -> bool:
        try:
            # Validate runtime dependency availability once.
            return importlib.util.find_spec("unstructured.partition.pdf") is not None
        except Exception:
            return False

    def load_and_process_documents(
        self,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> Dict[str, Any]:
        docs = []
        failed_files: List[str] = []

        def emit(stage: str, current: int, total: int, detail: str = "") -> None:
            if progress_callback:
                try:
                    progress_callback(stage, current, total, detail)
                except Exception:
                    pass

        pdf_root = Path(self.pdf_dir)
        if not pdf_root.exists():
            return {
                "ok": False,
                "message": f"PDF directory not found: {self.pdf_dir}",
                "pdf_count": 0,
                "loaded_docs": 0,
                "chunks": 0,
                "failed_files": [],
            }

        pdf_files = sorted(pdf_root.glob("*.pdf"))
        total = len(pdf_files)
        emit("prepare", 0, max(total, 1), "Preparing embeddings")
        self._get_embeddings()

        for i, pdf_file in enumerate(pdf_files, start=1):
            emit("loading", i - 1, max(total, 1), pdf_file.name)
            loaded_docs = self._load_pdf_with_fallback(str(pdf_file))
            if not loaded_docs:
                failed_files.append(pdf_file.name)
                emit("loading", i, max(total, 1), pdf_file.name)
                continue
            for d in loaded_docs:
                meta = d.metadata or {}
                meta["source"] = pdf_file.name
                page = meta.get("page", meta.get("page_number"))
                if isinstance(page, int):
                    # normalize to 0-based page for consistency
                    meta["page"] = max(0, page - 1) if page > 0 else page
                d.metadata = meta
            docs.extend(loaded_docs)
            emit("loading", i, max(total, 1), pdf_file.name)

        if not docs:
            return {
                "ok": False,
                "message": "No documents were loaded during indexing.",
                "pdf_count": len(pdf_files),
                "loaded_docs": 0,
                "chunks": 0,
                "failed_files": failed_files,
            }

        table_docs = [d for d in docs if (d.metadata or {}).get("category") == "Table"]
        non_table_docs = [d for d in docs if d not in table_docs]
        emit("chunking", 0, 1, "Chunking documents")

        splitter = RecursiveCharacterTextSplitter(
            # Character-based approximation of ~500-800 token chunks.
            chunk_size=800,
            chunk_overlap=120,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(non_table_docs)
        chunks.extend(table_docs)
        emit("embedding", 0, 1, "Building FAISS index")

        self.vector_store = FAISS.from_documents(chunks, self._get_embeddings())
        self.vector_store.save_local(self.vector_db_path)
        _VECTOR_STORE_CACHE[os.path.abspath(self.vector_db_path)] = self.vector_store
        emit("done", 1, 1, "Indexing complete")

        return {
            "ok": True,
            "message": "Index built successfully.",
            "pdf_count": len(pdf_files),
            "loaded_docs": len(docs),
            "chunks": len(chunks),
            "failed_files": failed_files,
        }

    def load_vector_store(self) -> None:
        cache_key = os.path.abspath(self.vector_db_path)
        if cache_key in _VECTOR_STORE_CACHE:
            self.vector_store = _VECTOR_STORE_CACHE[cache_key]
            return

        if os.path.exists(self.vector_db_path):
            try:
                self.vector_store = FAISS.load_local(
                    self.vector_db_path,
                    self._get_embeddings(),
                    allow_dangerous_deserialization=True,
                )
                _VECTOR_STORE_CACHE[cache_key] = self.vector_store
            except Exception:
                # Keep app alive on startup; user can still rebuild index.
                self.vector_store = None

    def retrieve_docs(self, question: str, k: int = 10) -> List[Any]:
        if not self.vector_store:
            self.load_vector_store()
        if not self.vector_store:
            return []

        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(question)

        source_hint = self._extract_source_hint(question)
        if source_hint:
            docs = self._filter_docs_by_source(docs, source_hint)
        return docs

    def format_docs(self, docs: List[Any]) -> str:
        blocks = []
        for d in docs:
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", None)
            if isinstance(page, int):
                blocks.append(f"Source: {source} (page {page + 1})\n{d.page_content}")
            else:
                blocks.append(f"Source: {source}\n{d.page_content}")
        return "\n\n".join(blocks)

    def answer_question(self, question: str) -> Dict[str, Any]:
        if self.llm is None:
            return {"answer": "LLM is not initialized.", "sources": []}

        docs = self.retrieve_docs(question, k=10)
        if not docs:
            return {"answer": "Not found in the provided SOPs.", "sources": []}

        context = self.format_docs(docs)
        if not context:
            return {"answer": "Not found in the provided SOPs.", "sources": []}

        if len(context) > 22000:
            context = context[:22000]

        prompt = PromptTemplate.from_template(
            """You are an expert Quality SOP assistant.
Use ONLY the provided PDF context.

Instructions:
- Give a detailed, clear, easy-to-understand answer.
- Use sections or bullets when helpful.
- Include only facts present in context.
- If context does not answer the question, output exactly: Not found in the provided SOPs.

Context:
{context}

Question: {question}

Answer:"""
        )

        rag_chain = (
            {"context": RunnableLambda(lambda _: context), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)
        sources = self._build_source_labels(docs)
        if self._is_low_quality(answer):
            fallback = self._extractive_fallback_answer(question, docs)
            answer = fallback or "Not found in the provided SOPs."

        # Keep responses clean: no quoted reference lines from chunks.
        # Sources should contain only document name and page number.
        return {"answer": answer, "sources": sources}

    def _build_source_labels(self, docs: List[Any]) -> List[str]:
        labels: List[str] = []
        seen = set()
        for d in docs[:10]:
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", None)
            label = f"{source} (page {page + 1})" if isinstance(page, int) else source
            if label in seen:
                continue
            seen.add(label)
            labels.append(label)
        return labels

    def _load_pdf_with_fallback(self, file_path: str) -> List[Any]:
        if self._can_use_unstructured:
            try:
                loader = UnstructuredPDFLoader(
                    file_path,
                    mode="elements",
                    strategy="hi_res",
                    infer_table_structure=True,
                    languages=["eng"],
                )
                return loader.load()
            except Exception:
                pass

        try:
            return PyMuPDFLoader(file_path).load()
        except Exception:
            try:
                return PyPDFLoader(file_path).load()
            except Exception:
                return []

    def _normalize_text(self, text: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").lower())
        return " ".join(cleaned.split())

    def _extract_source_hint(self, question: str) -> Optional[str]:
        if not self.vector_store:
            return None

        qn = self._normalize_text(question)
        if not qn:
            return None

        best = None
        best_len = 0
        try:
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = self.vector_store.docstore._dict.get(doc_id)
                if not doc:
                    continue
                source = doc.metadata.get("source", "")
                stem_norm = self._normalize_text(Path(source).stem)
                if stem_norm and stem_norm in qn and len(stem_norm) > best_len:
                    best = source
                    best_len = len(stem_norm)
        except Exception:
            return None
        return best

    def _filter_docs_by_source(self, docs: List[Any], source_hint: str) -> List[Any]:
        hint = source_hint.lower()
        filtered = [d for d in docs if hint in d.metadata.get("source", "").lower()]
        return filtered or docs

    def _keyword_set(self, question: str) -> List[str]:
        words = re.findall(r"[a-zA-Z]{3,}", (question or "").lower())
        stop = {"what", "which", "when", "where", "from", "with", "about", "into", "your", "this", "that", "sop"}
        return [w for w in words if w not in stop]

    def _pick_supporting_line(self, content: str, keywords: List[str]) -> str:
        lines = [ln.strip() for ln in re.split(r"[\n\.]+", content or "") if ln.strip()]
        if not lines:
            return ""
        if not keywords:
            return lines[0][:240]

        scored: List[Tuple[int, str]] = []
        for ln in lines:
            low = ln.lower()
            score = sum(1 for k in keywords if k in low)
            if score > 0:
                scored.append((score, ln))
        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored[0][1][:260]
        return lines[0][:240]

    def _build_references(self, docs: List[Any], question: str) -> List[str]:
        kws = self._keyword_set(question)
        refs = []
        seen = set()
        for d in docs[:6]:
            source = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page", None)
            source_ref = f"{source} (page {page + 1})" if isinstance(page, int) else source
            line = self._pick_supporting_line(d.page_content, kws)
            key = (source_ref, line)
            if key in seen:
                continue
            seen.add(key)
            if line:
                refs.append(f"- {source_ref}: \"{line}\"")
            else:
                refs.append(f"- {source_ref}")
        return refs

    def _is_low_quality(self, answer: str) -> bool:
        if not answer:
            return True
        t = answer.strip().lower()
        return len(t) < 20 or t in {"yes", "no", "yes.", "no.", "i don't know", "idk"}

    def _extractive_fallback_answer(self, question: str, docs: List[Any]) -> str:
        q = (question or "").strip().lower()
        if not docs:
            return ""

        # Handle common acronym-definition queries with grounded evidence.
        if (
            "sop" in q
            and any(k in q for k in ["what is", "define", "definition", "meaning", "full form"])
        ):
            for d in docs[:10]:
                lines = [ln.strip() for ln in re.split(r"[\n\.]+", d.page_content or "") if ln.strip()]
                for ln in lines:
                    if "standard operating procedure" in ln.lower():
                        return "SOP stands for Standard Operating Procedure."

            examples = []
            seen = set()
            for d in docs:
                src = d.metadata.get("source", "")
                stem = Path(src).stem.replace("_", " ").strip()
                if stem and stem not in seen:
                    seen.add(stem)
                    examples.append(stem)
                if len(examples) >= 3:
                    break

            if examples:
                joined = "; ".join(examples)
                return (
                    "In the provided PDFs, SOP is used as the prefix for procedure documents. "
                    f"Examples include: {joined}."
                )

        kws = self._keyword_set(question)
        for d in docs[:6]:
            line = self._pick_supporting_line(d.page_content, kws)
            if line:
                return f"Based on the SOP documents: {line}"
        return ""
