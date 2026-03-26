import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO

from dotenv import load_dotenv

# Load .env from the project root (one level up from src/)
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import requests

# Cache FAISS index in-process so Streamlit reruns don't reload from disk
_VECTOR_STORE_CACHE: Dict[str, FAISS] = {}
_DONUT_MODEL_CACHE: Dict[str, Any] = {}
_FLOWCHART_IMAGE_CACHE: Dict[Tuple[str, str], List[bytes]] = {}

class SOPRagPipeline:
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
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "")
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.groq_model = groq_model or os.getenv("GROQ_MODEL", "")
        self.groq_base_url = groq_base_url or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        self.llm = None

    def _setup_llm(self):
        """Sets up an LLM. Defaults to Groq if GROQ_API_KEY is set, else local."""
        provider = (self.llm_provider or "").lower().strip()
        if provider == "local":
            return self._setup_local_llm()
        if provider == "groq" or self.groq_api_key:
            return self._setup_groq_llm()
        return self._setup_local_llm()

    def _setup_local_llm(self):
        """Sets up a local LLM using HuggingFace."""
        model_id = "MBZUAI/LaMini-T5-738M"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, torch_dtype=torch.float32)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                if hasattr(prompt, "to_string"):
                    prompt = prompt.to_string()
                else:
                    prompt = str(prompt)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )
            return tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return RunnableLambda(generate_text)

    def _setup_groq_llm(self):
        """Sets up Groq via OpenAI-compatible Chat Completions."""
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is required for Groq API usage.")
        if not self.groq_model:
            raise ValueError("GROQ_MODEL is required for Groq API usage.")

        base_url = self.groq_base_url.rstrip("/")
        endpoint = f"{base_url}/chat/completions"

        def generate_text(prompt) -> str:
            if not isinstance(prompt, str):
                if hasattr(prompt, "to_string"):
                    prompt = prompt.to_string()
                else:
                    prompt = str(prompt)

            payload = {
                "model": self.groq_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0,
                "max_tokens": 1024,
            }
            headers = {
                "Authorization": f"Bearer {self.groq_api_key}",
                "Content-Type": "application/json",
            }

            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=60)
            except requests.exceptions.SSLError as exc:
                raise RuntimeError(
                    "Groq SSL connection failed. Please check your network, proxy, or TLS inspection settings and try again."
                ) from exc
            except requests.exceptions.Timeout as exc:
                raise RuntimeError("Groq request timed out. Please try again.") from exc
            except requests.exceptions.RequestException as exc:
                raise RuntimeError(f"Groq network error: {exc}") from exc

            if response.status_code >= 400:
                raise RuntimeError(f"Groq API error {response.status_code}: {response.text}")

            data = response.json()
            return data["choices"][0]["message"]["content"]

        return RunnableLambda(generate_text)


    def load_and_process_documents(self):
        """Loads PDFs, chunks them, and creates/saves the vector store."""
        documents = []
        if not os.path.exists(self.pdf_dir):
            print(f"Error: Directory {self.pdf_dir} not found.")
            return

        pdf_files = [f for f in os.listdir(self.pdf_dir) if f.endswith('.pdf')]
        
        print(f"Loading {len(pdf_files)} PDF documents...")
        for pdf_file in pdf_files:
            try:
                file_path = os.path.join(self.pdf_dir, pdf_file)
                try:
                    loader = UnstructuredPDFLoader(
                        file_path,
                        mode="elements",
                        strategy="fast",
                        languages=["eng"],
                    )
                    docs = loader.load()
                except Exception as exc:
                    # Fallback when Unstructured or Poppler is unavailable
                    msg = str(exc).lower()
                    if "unstructured" in msg or "poppler" in msg or "page count" in msg:
                        loader = PyPDFLoader(file_path)
                        docs = loader.load()
                    else:
                        raise
                for doc in docs:
                    doc.metadata["source"] = pdf_file
                documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_file}: {e}")

        if not documents:
            print("No documents were loaded.")
            return

        # Text Chunking (keep tables intact)
        table_docs = [d for d in documents if (d.metadata or {}).get("category") == "Table"]
        non_table_docs = [d for d in documents if d not in table_docs]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(non_table_docs)
        chunks.extend(table_docs)
        print(f"Created {len(chunks)} text chunks.")

        # Vector Store Creation
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.vector_db_path)
        _VECTOR_STORE_CACHE[os.path.abspath(self.vector_db_path)] = self.vector_store
        print(f"Vector store saved to {self.vector_db_path}")

    def load_vector_store(self):
        """Loads an existing vector store."""
        cache_key = os.path.abspath(self.vector_db_path)
        if cache_key in _VECTOR_STORE_CACHE:
            self.vector_store = _VECTOR_STORE_CACHE[cache_key]
            print("Vector store loaded from cache.")
            return

        if os.path.exists(self.vector_db_path):
            self.vector_store = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            _VECTOR_STORE_CACHE[cache_key] = self.vector_store
            print("Vector store loaded successfully.")
        else:
            print("Vector store not found. Please run indexing first.")

    def format_docs(self, docs):
        formatted = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            if page is not None:
                formatted.append(f"Source: {source} (page {page + 1})\n{doc.page_content}")
            else:
                formatted.append(f"Source: {source}\n{doc.page_content}")
        return "\n\n".join(formatted)

    def _normalize_text(self, text: str) -> str:
        lowered = (text or "").lower()
        cleaned = []
        for ch in lowered:
            if ch.isalnum() or ch.isspace():
                cleaned.append(ch)
            else:
                cleaned.append(" ")
        cleaned = "".join(cleaned)
        cleaned = cleaned.replace(" sop ", " ")
        cleaned = " ".join(cleaned.split())
        return cleaned

    def route_query(self, question: str) -> str:
        """Route query to text / table / flowchart."""
        q = (question or "").lower()
        if re.search(r"\b(flow\s*chart|flow\s*chat|process\s*flow|process\s*diagram|workflow\s*diagram|flow\s*diagram)\b", q):
            return "flowchart"
        if re.search(r"\b(table|tabular|raci|sipoc|matrix|list\s*all|summarize\s*in\s*table)\b", q):
            return "table"
        return "text"

    def _load_donut_components(self):
        model_name = os.getenv("DONUT_MODEL_NAME", "naver-clova-ix/donut-base-finetuned-docvqa").strip()
        if model_name in _DONUT_MODEL_CACHE:
            return _DONUT_MODEL_CACHE[model_name]
        try:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            print(f"[DONUT] Loading model: {model_name}")
            processor = DonutProcessor.from_pretrained(model_name)
            model = VisionEncoderDecoderModel.from_pretrained(model_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[DONUT] Model loaded successfully on device: {device}")
            model.to(device)
            model.eval()
            _DONUT_MODEL_CACHE[model_name] = (processor, model, device)
            return _DONUT_MODEL_CACHE[model_name]
        except Exception as e:
            print(f"[DONUT] Error loading Donut model: {type(e).__name__}: {e}")
            print(f"[DONUT] Ensure transformers and torch are installed properly")
            _DONUT_MODEL_CACHE[model_name] = None
            return None

    def _donut_answer(self, image, question: str) -> str:
        components = self._load_donut_components()
        if not components:
            print(f"[DONUT] Donut components failed to load.")
            return ""
        processor, model, device = components
        try:
            task_prompt = f"<s_docvqa><s_question>{question}</s_question><s_answer>"
            pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
            decoder_input_ids = processor.tokenizer(
                task_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.to(device)
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values,
                    decoder_input_ids=decoder_input_ids,
                    max_length=model.decoder.config.max_position_embeddings,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    num_beams=1,
                    bad_words_ids=[[processor.tokenizer.unk_token_id]],
                    return_dict_in_generate=True,
                )
            decoded = processor.batch_decode(outputs.sequences)[0]
            decoded = decoded.replace(processor.tokenizer.eos_token, "")
            decoded = decoded.replace(processor.tokenizer.pad_token, "")
            decoded = re.sub(r"<.*?>", " ", decoded)
            return " ".join(decoded.split()).strip()
        except Exception as e:
            print(f"[DONUT] Error during inference: {type(e).__name__}: {e}")
            return ""

    def _is_flowchart_image(self, image) -> bool:
        """Detect if a PIL image contains a flowchart using shape analysis.
        LENIENT - accepts valid flowcharts even with minimal graphics."""
        try:
            import cv2
            import numpy as np
            
            # Convert PIL image to opencv format
            image_array = np.array(image)
            if len(image_array.shape) == 3:
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                bgr_image = cv2.cvtColor(cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR), cv2.COLOR_RGB2BGR)
            
            # Convert to grayscale and apply edge detection
            gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Detect lines
            lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
            if lines is None:
                lines = []
            
            # Count horizontal vs vertical lines to detect TABLE GRIDS
            horizontal_lines = 0
            vertical_lines = 0
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                dy = abs(y2 - y1)
                dx = abs(x2 - x1)
                
                if dy < dx * 0.1:
                    horizontal_lines += 1
                elif dx < dy * 0.1:
                    vertical_lines += 1
            
            # REJECT: Pure table grid patterns
            if horizontal_lines > 10 and vertical_lines > 5:
                print(f"[FLOWCHART] Image: TABLE GRID ({horizontal_lines}H x {vertical_lines}V) -> REJECT")
                return False
            
            # Count shapes
            significant_contours = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                # Count substantial shapes
                if 300 < area < (image_array.shape[0] * image_array.shape[1] * 0.9):
                    significant_contours += 1
            
            # Flowchart score
            score = 0
            
            # More shapes = higher score
            if significant_contours >= 3:
                score += 2
            elif significant_contours >= 1:
                score += 1
            
            # Any connecting lines = flowchart indicator
            if len(lines) >= 2:
                score += 1
            
            # More lines = more complex flowchart
            if len(lines) >= 5:
                score += 1
            
            # Prefer diagrams that aren't pure grids
            if not (horizontal_lines > 5 and vertical_lines > 3):
                score += 1
            
            is_flowchart = score >= 2  # Lower threshold
            print(f"[FLOWCHART] Image: shapes={significant_contours}, lines={len(lines)}, h={horizontal_lines}v={vertical_lines}, score={score} -> {is_flowchart}")
            return is_flowchart
            
        except Exception as e:
            print(f"[FLOWCHART] Error in validation: {e}")
            return False
    
    def _extract_flowchart_images_fast(self, pdf_filename: str, question: str, max_images: int = 1) -> List[bytes]:
        """Extract flowchart images using fast text-based detection (no Donut)."""
        cache_key = (pdf_filename, self._normalize_text(question))
        if cache_key in _FLOWCHART_IMAGE_CACHE:
            print(f"[FLOWCHART] Cache hit for {pdf_filename}")
            return _FLOWCHART_IMAGE_CACHE[cache_key]

        try:
            import pdfplumber
        except Exception as e:
            print(f"[FLOWCHART] Error: pdfplumber not available. Install with: pip install pdfplumber. Details: {e}")
            return []

        pdf_path = Path(self.pdf_dir) / pdf_filename
        if not pdf_path.exists():
            print(f"[FLOWCHART] Error: PDF file not found at {pdf_path}")
            return []
        
        print(f"[FLOWCHART] Extracting from {pdf_filename}")

        images: List[Tuple[int, bytes]] = []
        
        # MUST have these keywords to be considered a flowchart
        required_keywords = {
            "flowchart", "flow chart", "process flow", "workflow",
            "workflow diagram", "process diagram", "process step diagram",
            "grn", "gated receipt", "receipt process", "goods receipt",
            "inventory process", "procurement process", "receipt note"
        }
        
        # MUST NOT have these keywords (definitely a table/matrix)
        reject_keywords = {
            "change history", "revision history", "raci", "approval matrix",
            "responsibility", "accountable", "consulted", "informed",
            "document control", "version control", "checklist", "summary table",
            "table of contents", "document version"
        }
        
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                total_pages = len(pdf.pages)
                print(f"[FLOWCHART] Scanning {total_pages} pages...")
                page_num = 0
                
                for page in pdf.pages:
                    page_num += 1
                    
                    # Extract page text
                    try:
                        page_text = (page.extract_text() or "").lower()
                    except Exception:
                        page_text = ""
                    
                    # MUST have flowchart keyword
                    has_flowchart_keyword = any(kw in page_text for kw in required_keywords)
                    if not has_flowchart_keyword:
                        continue
                    
                    # MUST NOT have table/matrix keywords
                    has_reject_keyword = any(kw in page_text for kw in reject_keywords)
                    if has_reject_keyword:
                        print(f"[FLOWCHART] Page {page_num}: Rejected (contains table marker)")
                        continue
                    
                    matched_kw = [kw for kw in required_keywords if kw in page_text][0]
                    print(f"[FLOWCHART] Page {page_num}: Keyword match - '{matched_kw}'. Validating image...")
                    
                    # Validate the image to confirm it's actually a flowchart
                    try:
                        page_image = page.to_image(resolution=150).original
                    except Exception as e:
                        print(f"[FLOWCHART] Page {page_num}: Error converting to image: {e}")
                        continue
                    
                    # Check if image contains flowchart patterns
                    if not self._is_flowchart_image(page_image):
                        print(f"[FLOWCHART] Page {page_num}: [REJECT] Image validation FAILED (not a real flowchart)")
                        continue
                    
                    print(f"[FLOWCHART] Page {page_num}: [OK] VALIDATED - Extracting flowchart image")
                    score = 10

                    words = page.extract_words(
                        x_tolerance=2,
                        y_tolerance=2,
                        keep_blank_chars=False,
                        use_text_flow=True,
                    ) or []
                    if not words:
                        continue

                    heading_bottom = page.height * 0.12
                    heading_candidates = [
                        float(w.get("bottom", 0))
                        for w in words
                        if (w.get("text") or "").lower() in {"flow", "flowchart", "chart", "process", "workflow", "grn"}
                    ]
                    if heading_candidates:
                        heading_bottom = max(heading_candidates) + 8

                    stop_markers = {"raci", "sipoc", "kpi", "roles", "responsibility", "reference", "summary", "footer", "page"}
                    stop_top = page.height - 20
                    for word in words:
                        token = (word.get("text") or "").lower()
                        word_top = float(word.get("top", page.height))
                        if word_top <= heading_bottom + 12:
                            continue
                        if token in stop_markers:
                            stop_top = min(stop_top, word_top - 8)

                    # Extract graphic elements (lines, boxes, curves - indicators of flowchart)
                    graphic_boxes = []
                    for collection in (page.lines, page.rects, page.curves):
                        for obj in collection:
                            x0 = float(obj.get("x0", 0))
                            x1 = float(obj.get("x1", page.width))
                            top = float(obj.get("top", 0))
                            bottom = float(obj.get("bottom", page.height))
                            width = max(0.0, x1 - x0)
                            height = max(0.0, bottom - top)
                            if top < heading_bottom or bottom > stop_top:
                                continue
                            if width < 20 and height < 20:
                                continue
                            # Skip full-width lines (likely table separators)
                            if width >= page.width * 0.92 and height <= 5:
                                continue
                            graphic_boxes.append((x0, top, x1, bottom))
                    
                    # If no graphics found, skip (likely not a flowchart)
                    if not graphic_boxes:
                        print(f"[FLOWCHART] Page {page_num}: No graphic elements found, skipping")
                        continue
                    
                    # Check if it's a table pattern (many horizontal lines at regular intervals = table grid)
                    horizontal_lines = [obj for obj in page.lines if float(obj.get("height", 0)) < 2]
                    if len(horizontal_lines) > 15:  # Many horizontal lines = likely a table
                        print(f"[FLOWCHART] Page {page_num}: Detected table pattern ({len(horizontal_lines)} horizontal lines), skipping")
                        continue

                    text_boxes = []
                    for word in words:
                        top = float(word.get("top", 0))
                        bottom = float(word.get("bottom", 0))
                        if top < heading_bottom or bottom > stop_top:
                            continue
                        text_boxes.append((
                            float(word.get("x0", 0)),
                            top,
                            float(word.get("x1", page.width)),
                            bottom,
                        ))

                    if graphic_boxes:
                        x0 = min(box[0] for box in graphic_boxes)
                        top = min(box[1] for box in graphic_boxes)
                        x1 = max(box[2] for box in graphic_boxes)
                        bottom = max(box[3] for box in graphic_boxes)
                        for tx0, ttop, tx1, tbottom in text_boxes:
                            if tx1 < x0 - 80 or tx0 > x1 + 80:
                                continue
                            if tbottom < top - 40 or ttop > bottom + 40:
                                continue
                            x0 = min(x0, tx0)
                            top = min(top, ttop)
                            x1 = max(x1, tx1)
                            bottom = max(bottom, tbottom)
                    elif text_boxes:
                        x0 = min(box[0] for box in text_boxes)
                        top = min(box[1] for box in text_boxes)
                        x1 = max(box[2] for box in text_boxes)
                        bottom = max(box[3] for box in text_boxes)
                    else:
                        continue

                    # Tighter padding for flowcharts (reduce excess margin)
                    pad_x = max(8.0, page.width * 0.01)
                    pad_y = max(8.0, page.height * 0.01)
                    crop_box = (
                        max(0.0, x0 - pad_x),
                        max(heading_bottom, top - pad_y),
                        min(page.width, x1 + pad_x),
                        min(stop_top, bottom + pad_y),
                    )
                    
                    # Only expand to full width if content is very narrow
                    crop_width = crop_box[2] - crop_box[0]
                    if crop_width < page.width * 0.2:
                        # Keep narrow width for focused diagrams
                        pass
                    
                    min_height = page.height * 0.12
                    if crop_box[3] - crop_box[1] < min_height:
                        continue

                    # Extract and render the cropped region
                    try:
                        cropped_page = page.crop(crop_box)
                        rendered = cropped_page.to_image(resolution=220).original
                        buffer = BytesIO()
                        rendered.save(buffer, format="PNG")
                        print(f"[FLOWCHART] Page {page_num}: Extracted flowchart ({rendered.size[0]}x{rendered.size[1]}px)")
                        images.append((score, buffer.getvalue()))
                    except Exception as e:
                        print(f"[FLOWCHART] Page {page_num}: Error rendering: {e}")
                        continue

            images.sort(key=lambda item: item[0], reverse=True)
            if images:
                print(f"[FLOWCHART] [OK] Found {len(images)} flowchart image(s)")
            else:
                print(f"[FLOWCHART] No flowchart pages found in {pdf_filename}")
                
            result = [image for _, image in images[:max_images]]
            _FLOWCHART_IMAGE_CACHE[cache_key] = result
            return result
        except Exception as e:
            print(f"[FLOWCHART] Error: {type(e).__name__}: {e}")
            import traceback
            print(f"[FLOWCHART] Traceback: {traceback.format_exc()}")
            return []

    def _extract_source_hint(self, question: str) -> Optional[str]:
        if not self.vector_store:
            return None

        question_norm = self._normalize_text(question)
        if not question_norm:
            return None

        sources = set()
        try:
            for doc_id in self.vector_store.index_to_docstore_id.values():
                doc = self.vector_store.docstore._dict.get(doc_id)
                if doc is None:
                    continue
                source = doc.metadata.get("source", "")
                if source:
                    sources.add(source)
        except Exception:
            return None

        best_source = None
        best_len = 0
        for source in sources:
            name = os.path.splitext(source)[0]
            name_norm = self._normalize_text(name)
            if not name_norm:
                continue
            if name_norm in question_norm and len(name_norm) > best_len:
                best_source = source
                best_len = len(name_norm)

        return best_source

    def _filter_docs_by_source(self, docs: List, source_hint: Optional[str]) -> List:
        if not source_hint:
            return docs
        hint_lower = source_hint.lower()
        filtered = [doc for doc in docs if hint_lower in doc.metadata.get("source", "").lower()]
        return filtered or docs

    def _match_pdf_file(self, question: str) -> Optional[str]:
        """Match a PDF filename from the user's question using fuzzy name matching."""
        question_norm = self._normalize_text(question)
        if not question_norm:
            return None

        pdf_dir = Path(self.pdf_dir)
        if not pdf_dir.exists():
            return None

        best_match = None
        best_len = 0
        for pdf in pdf_dir.glob("*.pdf"):
            name_norm = self._normalize_text(pdf.stem)
            if not name_norm:
                continue
            # Check if the normalized PDF name appears in the question
            if name_norm in question_norm and len(name_norm) > best_len:
                best_match = pdf.name
                best_len = len(name_norm)

            # Also check individual significant words from the PDF name
            name_words = set(name_norm.split())
            # Remove very common/short words
            name_words -= {"sop", "ut", "of", "and", "the", "in", "for", "a", "an", "to"}
            question_words = set(question_norm.split())
            overlap = name_words & question_words
            # If most significant words match, it's the right PDF
            if len(name_words) > 0 and len(overlap) >= max(2, len(name_words) * 0.5):
                score = len(overlap)
                if score > best_len:
                    best_match = pdf.name
                    best_len = score

        return best_match

    def _get_targeted_pdf_text(self, pdf_filename: str, question: str = "", max_chars: int = 20000) -> str:
        """Read text content of a PDF file. Selects most relevant pages if it exceeds max_chars."""
        from pypdf import PdfReader
        import re

        pdf_path = Path(self.pdf_dir) / pdf_filename
        if not pdf_path.exists():
            return ""

        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append((i, text))
            
            # If total length is within limits, return everything
            total_len = sum(len(text) for _, text in pages)
            if total_len <= max_chars:
                return "\n\n".join([f"--- {pdf_filename} (page {i + 1}) ---\n{text}" for i, text in pages])

            # Normalize query words, removing common stop words
            q_words = set(re.findall(r'\w+', question.lower())) - {"sop", "the", "a", "an", "of", "and", "in", "to", "for", "from", "flow", "chart", "table", "process", "what", "is", "diagram"}
            
            # Score each page
            page_scores = []
            for idx, (i, text) in enumerate(pages):
                text_lower = text.lower()
                # 1 point per matching keyword
                score = sum(3 for w in q_words if w in text_lower)
                # Boost if it contains exactly "process description" (crucial for flowcharts)
                if "process description" in text_lower:
                    score += 15
                page_scores.append((score, idx, i, text))
            
            # Sort by score descending
            page_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Select top pages until we hit max_chars.
            selected_indices = set()
            current_chars = 0
            
            for score, idx, i, text in page_scores:
                # Format page text
                page_text = f"--- {pdf_filename} (page {i + 1}) ---\n{text}"
                if current_chars + len(page_text) > max_chars and selected_indices:
                    break # Stop if we can't fit this page and we already have some
                
                selected_indices.add(idx)
                current_chars += len(page_text)
                
                # Try to add the next adjoining page if it fits (context spill over)
                if idx + 1 < len(pages) and idx + 1 not in selected_indices:
                    next_text = pages[idx+1][1]
                    next_page_text = f"--- {pdf_filename} (page {pages[idx+1][0] + 1}) ---\n{next_text}"
                    if current_chars + len(next_page_text) <= max_chars:
                        selected_indices.add(idx + 1)
                        current_chars += len(next_page_text)
            
            # Re-sort selected pages by their original order (idx)
            selected_pages = [pages[idx] for idx in sorted(list(selected_indices))]
            return "\n\n".join([f"--- {pdf_filename} (page {i + 1}) ---\n{text}" for i, text in selected_pages])
        except Exception:
            return ""

    def _get_pdf_pages_with_keywords(self, pdf_filename: str, keywords: List[str], max_chars: int = 30000) -> str:
        """Collect pages containing any keyword, plus adjacent pages for continuation."""
        from pypdf import PdfReader

        pdf_path = Path(self.pdf_dir) / pdf_filename
        if not pdf_path.exists():
            return ""

        try:
            reader = PdfReader(str(pdf_path))
            pages = []
            selected_indices = set()
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                text_lower = text.lower()
                if any(k in text_lower for k in keywords):
                    selected_indices.add(i)
                    # include adjacent pages to capture table continuations
                    if i - 1 >= 0:
                        selected_indices.add(i - 1)
                    if i + 1 < len(reader.pages):
                        selected_indices.add(i + 1)

            if not selected_indices:
                return ""

            for i in sorted(selected_indices):
                text = reader.pages[i].extract_text() or ""
                if text.strip():
                    pages.append((i, text))

            joined = "\n\n".join([f"--- {pdf_filename} (page {i + 1}) ---\n{text}" for i, text in pages])
            if len(joined) > max_chars:
                return joined[:max_chars]
            return joined
        except Exception:
            return ""

    def _get_pdf_pages_with_keywords_layout(self, pdf_filename: str, keywords: List[str], max_chars: int = 40000) -> str:
        """Collect pages using layout text to preserve columns."""
        from pypdf import PdfReader

        pdf_path = Path(self.pdf_dir) / pdf_filename
        if not pdf_path.exists():
            return ""

        try:
            reader = PdfReader(str(pdf_path))
            selected_indices = set()
            for i, page in enumerate(reader.pages):
                text = page.extract_text(extraction_mode="layout") or ""
                if not text.strip():
                    continue
                text_lower = text.lower()
                if any(k in text_lower for k in keywords):
                    selected_indices.add(i)
                    if i - 1 >= 0:
                        selected_indices.add(i - 1)
                    if i + 1 < len(reader.pages):
                        selected_indices.add(i + 1)

            if not selected_indices:
                return ""

            pages = []
            for i in sorted(selected_indices):
                text = reader.pages[i].extract_text(extraction_mode="layout") or ""
                if text.strip():
                    pages.append((i, text))

            joined = "\n\n".join([f"--- {pdf_filename} (page {i + 1}) ---\n{text}" for i, text in pages])
            if len(joined) > max_chars:
                return joined[:max_chars]
            return joined
        except Exception:
            return ""

    def _extract_table_by_header(
        self,
        layout_text: str,
        headers: List[str],
        stop_markers: Optional[List[str]] = None,
        skip_markers: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Extract table rows using multi-space separation instead of fragile position-based extraction."""
        if not layout_text:
            return []

        stop_markers = [s.lower() for s in (stop_markers or [])]
        skip_markers = [s.lower() for s in (skip_markers or [])]
        lines = [ln.rstrip("\n") for ln in layout_text.splitlines()]
        header_idx = None
        headers_lower = [h.lower() for h in headers]
        for i, line in enumerate(lines):
            lower = line.lower()
            if all(h in lower for h in headers_lower):
                header_idx = i
                break

        if header_idx is None:
            return []

        rows = []
        for line in lines[header_idx + 1:]:
            lower = line.lower().strip()
            if not lower:
                continue
            if any(m in lower for m in stop_markers):
                break
            if any(m in lower for m in skip_markers):
                continue

            # Skip page separators
            if line.strip().startswith("---") and "page" in lower:
                continue

            # Split by 2+ spaces for better column detection
            parts = re.split(r"\s{2,}", line.strip())
            parts = [p.strip() for p in parts if p.strip()]
            
            if not parts:
                continue
            
            # Pad with N/A if needed, trim to header count
            while len(parts) < len(headers):
                parts.append("N/A")
            cols = parts[:len(headers)]
            
            rows.append([c.strip() if c.strip() else "N/A" for c in cols])

        return rows

    def _extract_table_rows(self, text: str, header_tokens: List[str], min_cols: int = 5) -> List[List[str]]:
        """Heuristic table parser using multi-space separation and row buffering."""
        if not text:
            return []

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        header_line_idx = None
        header_tokens_lower = [t.lower() for t in header_tokens]
        for i, line in enumerate(lines):
            lower = line.lower()
            if all(tok in lower for tok in header_tokens_lower):
                header_line_idx = i
                break

        if header_line_idx is None:
            return []

        rows = []
        buffer = []
        for line in lines[header_line_idx + 1:]:
            # Skip page separators
            if line.startswith("---") and "page" in line.lower():
                continue

            # Split on 2+ spaces to preserve multi-word cells
            parts = re.split(r"\s{2,}", line)
            parts = [p.strip() for p in parts if p.strip()]
            if not parts:
                continue

            if buffer:
                # Append continuation text to last cell if row is incomplete
                buffer[-1] = (buffer[-1] + " " + " ".join(parts)).strip()
            else:
                buffer.extend(parts)

            # If we have enough columns, finalize row
            if len(buffer) >= min_cols:
                row = buffer[:min_cols]
                rows.append(row)
                buffer = []

        return rows

    def _format_markdown_table(self, headers: List[str], rows: List[List[str]]) -> str:
        if not rows or not headers:
            return ""
        
        # Sanitize headers and rows to ensure no pipe characters break formatting
        safe_headers = [str(h).replace("|", "\\|").strip() for h in headers]
        safe_rows = []
        for row in rows:
            safe_row = [str(cell).replace("|", "\\|").strip() for cell in row]
            safe_rows.append(safe_row)
        
        header_line = "| " + " | ".join(safe_headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(safe_headers)) + " |"
        body = "\n".join("| " + " | ".join(r) + " |" for r in safe_rows)
        return "\n".join([header_line, sep_line, body])
    
    def _clean_table_output(self, markdown_text: str) -> str:
        """Clean markdown table output to prevent bleed between tables."""
        if not markdown_text:
            return ""
        
        # Ensure each table section is properly terminated
        lines = markdown_text.split("\n")
        cleaned = []
        in_table = False
        table_line_count = 0
        
        for line in lines:
            # Detect page markers
            if line.strip().startswith("---") and "page" in line.lower():
                if in_table and table_line_count > 0:
                    # End previous table properly
                    in_table = False
                    table_line_count = 0
                cleaned.append(line)
                continue
            
            # Detect table lines (pipes)
            if "|" in line:
                in_table = True
                table_line_count += 1
                cleaned.append(line)
            else:
                # If we were in a table and now see non-table content, add blank line
                if in_table and line.strip():
                    # Non-table line, close the table section
                    if table_line_count > 2:  # At least header + separator + 1 row
                        cleaned.append("")  # Blank line separator
                    in_table = False
                    table_line_count = 0
                
                if line.strip():  # Only keep non-empty non-table lines
                    cleaned.append(line)
        
        result = "\n".join(cleaned)
        # Remove multiple consecutive blank lines
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")
        
        return result.strip()

    def _remove_mostly_empty_columns(self, headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """Remove columns that are mostly empty - helps with multi-level header extraction."""
        if not headers or not rows:
            return headers, rows
        
        # Calculate fill percentage for each column
        col_stats = []
        for i in range(len(headers)):
            col_vals = [str(r[i]).strip() if i < len(r) else "" for r in rows]
            non_empty = len([v for v in col_vals if v and v != "N/A"])
            fill_pct = non_empty / len(rows) if rows else 0
            col_stats.append((i, fill_pct, headers[i].strip().lower()))
        
        # Keep columns that are:
        # 1. Critical headers (Process, Supplier, Customer, etc)
        # 2. More than 30% filled
        critical = {"process", "supplier", "input", "output", "customer", "responsible", 
                   "accountable", "consulted", "informed", "sl.", "sl", "no.", "no", "sl. no."}
        
        keep_indices = []
        for idx, fill_pct, header_name in col_stats:
            if header_name in critical:
                keep_indices.append(idx)
            elif fill_pct > 0.3:  # More than 30% filled
                keep_indices.append(idx)
        
        if not keep_indices:
            return headers, rows
        
        # Rebuild headers and rows with only kept columns
        new_headers = [headers[i] for i in keep_indices]
        new_rows = []
        for row in rows:
            new_row = [row[i] if i < len(row) else "" for i in keep_indices]
            new_rows.append(new_row)
        
        return new_headers, new_rows

    def _clean_table_text(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []

        for line in lines:
            line = re.sub(r"\s{2,}", " | ", line)
            cleaned.append(line)

        return "\n".join(cleaned)

    def _crop_page_for_tables(self, page, top_ratio: float = 0.04, bottom_ratio: float = 0.04):
        """Crop page to reduce header/footer noise before table extraction."""
        try:
            h = float(page.height)
            w = float(page.width)
        except Exception:
            return page
        top = max(0.0, h * top_ratio)
        bottom = max(top + 1.0, h * (1.0 - bottom_ratio))
        if bottom - top < h * 0.5:
            return page
        return page.crop((0, top, w, bottom))

    def _is_header_footer_text(self, text: str) -> bool:
        t = re.sub(r"\s+", " ", (text or "").strip().lower())
        if not t:
            return True
        patterns = [
            r"document title", r"document no", r"document number",
            r"document classification", r"document status",
            r"document template", r"effective date", r"next review",
            r"confidential", r"internal", r"classified",
            r"contents of this document", r"nda",
            r"page\s*\d+\s*of\s*\d+", r"\bpage\s*\d+\b",
            r"\bissue\b", r"\bversion\b",
            r"^-+\s*page\s*\d+", r"^page\s*\d+$",
            r"process management controls",
            r".*controls.*",
            r"^effective date:", r"^next review date:",
            r"^process:", r"^control:",
        ]
        return any(re.search(p, t) for p in patterns)

    def _is_metadata_row(self, row: List[str], headers: List[str]) -> bool:
        """Check if a row is metadata or section headers, not actual data."""
        if not row:
            return True
        
        # Join row text for analysis
        row_text = " ".join(str(x).strip().lower() for x in row)
        
        # Check for metadata labels
        metadata_patterns = [
            r"^effective date", r"^next review", r"^process:", r"^control:",
            r"process management", r"implementation", r"pilot.*implementation",
            r"full.scale implement", r"archive and sustain",
            r"deployment", r"deployment plan", r"deployment results",
            r"^risks?", r"^schedule", r"^success criteria",
            r"^a\)\s", r"^b\)\s",  # Lettered list items
        ]
        
        if any(re.search(p, row_text) for p in metadata_patterns):
            return True
        
        # Check if row has very few cells with actual content
        non_empty_cells = [c for c in row if str(c).strip() not in {"", "N/A"}]
        if len(non_empty_cells) == 1:  # Only one cell has content - likely metadata
            return True
        
        return False

    def _normalize_table(self, table: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        """Normalize headers/rows with conservative approach - avoid losing data."""
        if not table:
            return [], []

        headers = [str(h).strip() if h is not None else "N/A" for h in table[0]]
        rows = []
        max_cols = len(headers)
        
        for row in table[1:]:
            cleaned = []
            for i, cell in enumerate(row):
                val = str(cell).replace("\n", " ").strip() if cell else ""
                cleaned.append(val if val else "N/A")
            
            # Ensure consistent column count
            while len(cleaned) < max_cols:
                cleaned.append("N/A")
            cleaned = cleaned[:max_cols]  # Trim excess columns
            
            rows.append(cleaned)

        # Merge split "Sl." + "No." columns into "Sl. No." (conservative)
        if len(headers) >= 2:
            h0 = headers[0].lower().replace(".", "").replace(" ", "").strip()
            h1 = headers[1].lower().replace(".", "").replace(" ", "").strip()
            if h0 in {"sl", "s1"} and h1 in {"no", "no1"}:
                headers = ["Sl. No."] + headers[2:]
                merged_rows = []
                for r in rows:
                    sl = r[0] if len(r) > 0 else "N/A"
                    no = r[1] if len(r) > 1 else "N/A"
                    parts = [p for p in [sl, no] if str(p).strip() not in {"", "N/A"}]
                    combined = " ".join(parts).strip() if parts else "N/A"
                    merged_rows.append([combined] + r[2:])
                rows = merged_rows

        # Fix obvious N/A columns that leaked from extraction (but preserve data)
        i = 0
        while i < len(headers) - 1:
            h = headers[i].strip().lower()
            h_next = headers[i + 1].strip().lower()
            
            # Only merge if BOTH columns are N/A (not just header)
            col_vals = [r[i] if i < len(r) else "N/A" for r in rows]
            next_col_vals = [r[i + 1] if i + 1 < len(r) else "N/A" for r in rows]
            
            if h == "n/a" and h_next != "n/a" and all(v == "N/A" for v in col_vals):
                # Drop the N/A header column only if it's truly empty
                headers.pop(i)
                for r in rows:
                    if i < len(r):
                        r.pop(i)
                continue
            
            i += 1

        # Drop ONLY completely empty columns
        if headers:
            keep = []
            for i in range(len(headers)):
                h_norm = headers[i].strip().lower()
                col_vals = [headers[i]] + [r[i] if i < len(r) else "N/A" for r in rows]
                
                # Always keep important columns
                if h_norm in {"sl.", "sl", "no.", "no", "sl. no.", "process", "responsible", "accountable", "consulted", "informed"}:
                    keep.append(i)
                    continue
                
                # Drop only if truly empty
                if all(str(v).strip() in {"", "N/A"} for v in col_vals):
                    continue
                
                keep.append(i)
            
            headers = [headers[i] for i in keep]
            rows = [[(r[i] if i < len(r) else "N/A") for i in keep] for r in rows]

        # Remove mostly empty columns from multi-level header extraction
        headers, rows = self._remove_mostly_empty_columns(headers, rows)

        return headers, rows

    def _filter_table_rows(self, headers: List[str], rows: List[List[str]]) -> List[List[str]]:
        """Filter rows but preserve valid data - remove metadata and section headers."""
        filtered = []
        header_join = " ".join(h.lower() for h in headers)
        
        for row in rows:
            joined = " ".join(str(x).lower() for x in row)
            
            # Skip completely empty rows
            if not joined.strip() or all(str(x).strip() in {"", "N/A"} for x in row):
                continue
            
            # Skip if row is entirely the header repeated
            if header_join and header_join == joined:
                continue
            
            # Skip explicit page separators like "--- page 2 ---"
            if re.match(r"^-+\s+page\s+\d+\s+-+$", joined):
                continue
            
            # Filter out metadata rows and section headers
            if self._is_metadata_row(row, headers):
                continue
            
            # Skip if row is entirely header/footer pattern
            if self._is_header_footer_text(joined) and all(len(str(x).strip()) < 50 for x in row):
                continue
            
            filtered.append(row)
        
        return filtered

    def _table_keywords_from_question(self, question: str) -> List[str]:
        tokens = re.findall(r"[a-zA-Z]{3,}", (question or "").lower())
        stop = {
            "table", "tables", "show", "list", "give", "provide", "extract",
            "from", "the", "and", "for", "with", "that", "this", "these", "those",
            "process", "flow", "chart", "flowchart", "diagram", "overall",
            "matrix", "column", "columns", "row", "rows", "data", "details",
        }
        return [t for t in tokens if t not in stop]

    def _find_pdf_by_keywords(self, keywords: List[str], max_pages: int = 5) -> Optional[str]:
        """Scan PDFs in directory and return first that contains any keyword."""
        if not keywords:
            return None
        pdf_dir = Path(self.pdf_dir)
        if not pdf_dir.exists():
            return None
        try:
            from pypdf import PdfReader
        except Exception:
            return None

        kws = [k.lower() for k in keywords]
        for pdf in pdf_dir.glob("*.pdf"):
            try:
                reader = PdfReader(str(pdf))
                pages = reader.pages[:max_pages]
                for page in pages:
                    try:
                        text = page.extract_text(extraction_mode="layout") or ""
                    except Exception:
                        text = page.extract_text() or ""
                    lower = text.lower()
                    if any(k in lower for k in kws):
                        return pdf.name
            except Exception:
                continue
        return None

    def extract_tables_pdfplumber(self, pdf_path: str, required_keywords: Optional[List[str]] = None):
        import pdfplumber

        tables_text = []
        required_keywords = [k.lower() for k in (required_keywords or []) if k]

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page = self._crop_page_for_tables(page)
                try:
                    tables = page.extract_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                    })
                except Exception:
                    tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    # Normalize row lengths - use consistent column count
                    max_cols = max(len(row) for row in table if row)
                    normalized = []
                    for row in table:
                        new_row = []
                        for i in range(max_cols):
                            val = row[i] if i < len(row) else None
                            if val is None or str(val).strip() == "":
                                val = ""
                            new_row.append(str(val).strip())
                        normalized.append(new_row)

                    headers, rows = self._normalize_table(normalized)
                    if not headers or self._is_header_footer_text(" ".join(headers)):
                        continue
                    
                    # Apply improved filtering first
                    rows = self._filter_table_rows(headers, rows)
                    
                    # Additional filtering: remove rows where all cells are too short or metadata-like
                    filtered_rows = []
                    for row in rows:
                        # Check if this row has reasonable data distribution
                        non_empty = [c for c in row if str(c).strip() and str(c).strip() not in {"N/A", ""}]
                        
                        # If most cells are empty, skip this row
                        if len(non_empty) < len(row) * 0.3:  # Less than 30% filled
                            continue
                        
                        # If it's marked as metadata, skip it
                        if self._is_metadata_row(row, headers):
                            continue
                        
                        filtered_rows.append(row)
                    
                    if not filtered_rows:
                        continue

                    if required_keywords:
                        joined = " ".join(headers + [" ".join(r) for r in filtered_rows]).lower()
                        if not any(k in joined for k in required_keywords):
                            continue

                    md = self._format_markdown_table(headers, filtered_rows)
                    if md:
                        tables_text.append(f"--- Page {page_num+1} ---\n{md}")

        result = "\n\n".join(tables_text)
        return self._clean_table_output(result)

    def extract_tables_paddleocr(self, pdf_path: str) -> str:
        """Extract tables using PaddleOCR PP-Structure (layout-aware)."""
        try:
            from paddleocr import PPStructure, draw_structure_result  # type: ignore
        except Exception:
            return ""

        tables_text = []
        try:
            ocr_engine = PPStructure(show_log=False)
        except Exception:
            return ""

        # PP-Structure expects images; convert PDF pages to images via pdf2image
        try:
            from pdf2image import convert_from_path  # type: ignore
        except Exception:
            return ""

        try:
            pages = convert_from_path(pdf_path)
        except Exception:
            return ""

        for page_num, img in enumerate(pages, start=1):
            try:
                result = ocr_engine(img)
            except Exception:
                continue

            for block in result:
                if block.get("type") != "table":
                    continue
                html = block.get("res", {}).get("html", "")
                if not html:
                    continue
                # Convert HTML table to markdown using pandas
                try:
                    import pandas as pd  # type: ignore
                    dfs = pd.read_html(html)
                    for df in dfs:
                        md = df.to_markdown(index=False)
                        tables_text.append(f"--- Page {page_num} ---\n{md}")
                except Exception:
                    continue

        return "\n\n".join(tables_text)


    def extract_any_table_with_keywords(self, pdf_path: str, keywords: List[str]) -> str:
        """Generic table extractor for when specific extractors fail. Looks for tables with keywords."""
        import pdfplumber
        
        tables_text = []
        keywords_lower = [k.lower() for k in keywords]
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page = self._crop_page_for_tables(page)
                try:
                    tables = page.extract_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                    })
                except Exception:
                    tables = page.extract_tables()
                
                if not tables:
                    continue
                
                for table in tables:
                    if not table or len(table) < 2:
                        continue
                    
                    # Check if any keyword appears in the table
                    table_text = " ".join(str(cell) for row in table for cell in row).lower()
                    if not any(k in table_text for k in keywords_lower):
                        continue
                    
                    # Normalize and extract
                    raw_headers = [str(h).strip() if h else "" for h in table[0]]
                    headers, rows = self._normalize_table([raw_headers] + table[1:])
                    
                    if not headers:
                        continue
                    
                    # Light filtering - just remove completely empty rows
                    filtered_rows = []
                    for row in rows:
                        if any(str(c).strip() and str(c).strip() not in {"N/A", ""} for c in row):
                            filtered_rows.append(row)
                    
                    if not filtered_rows:
                        continue
                    
                    md = self._format_markdown_table(headers, filtered_rows)
                    if md:
                        tables_text.append(f"--- Page {page_num+1} ---\n{md}")
        
        result = "\n\n".join(tables_text)
        return self._clean_table_output(result)
        import pdfplumber

        final_tables = []
        current_headers = None
        current_rows = []
        current_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page = self._crop_page_for_tables(page)
                try:
                    tables = page.extract_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                    })
                except Exception:
                    tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    raw_headers = [str(h).strip() if h else "" for h in table[0]]

                    # Detect RACI table
                    header_text = " ".join(raw_headers).lower()
                    has_raci_header = all(k in header_text for k in ["responsible", "accountable", "consulted", "informed"])

                    rows = []
                    for row in table[1:]:
                        cleaned = []
                        for cell in row:
                            if cell:
                                cleaned.append(str(cell).replace("\n", " ").strip())
                            else:
                                cleaned.append("")

                        # Ensure consistent columns
                        while len(cleaned) < len(raw_headers):
                            cleaned.append("")

                        rows.append(cleaned)

                    # If first data row is a split header (e.g., "No." under "Sl.")
                    if raw_headers and rows:
                        first_row_text = " ".join(x.lower() for x in rows[0])
                        if ("no." in first_row_text or "no" in first_row_text) and all(
                            x in {"", "no", "no."} for x in [rows[0][0]] + rows[0][2:]
                        ):
                            raw_headers = [raw_headers[0] + " No."] + raw_headers[1:]
                            rows = rows[1:]

                    # If this looks like a continuation page, attach to current
                    if not has_raci_header and current_headers and rows:
                        # Trim or pad to match current header width
                        width = len(current_headers)
                        fixed_rows = []
                        for row in rows:
                            if len(row) < width:
                                row = row + [""] * (width - len(row))
                            elif len(row) > width:
                                row = row[:width]
                            fixed_rows.append(row)
                        # Only treat as continuation if it has numeric Sl/No in first column
                        looks_like_continuation = any(
                            re.match(r"^\d+\s*$", (r[0] or "").strip()) or
                            (len(r) > 1 and (r[1] or "").strip() not in {"", ""})
                            for r in fixed_rows
                        )
                        if looks_like_continuation:
                            fixed_rows = self._filter_table_rows(current_headers, fixed_rows)
                            current_rows.extend(fixed_rows)
                        if page_num + 1 not in current_pages:
                            current_pages.append(page_num + 1)
                        continue

                    if not has_raci_header:
                        continue

                    # Normalize headers/rows
                    headers, rows = self._normalize_table([raw_headers] + rows)
                    
                    if not headers or self._is_header_footer_text(" ".join(headers)):
                        continue

                    # Remove useless columns, but keep Sl./No. when present
                    desired_headers = {"sl.", "sl", "no.", "no", "sl. no.", "process", "responsible", "accountable", "consulted", "informed"}
                    keep_indices = []
                    for idx, h in enumerate(headers):
                        h_norm = h.strip().lower()
                        if h_norm in desired_headers:
                            keep_indices.append(idx)
                            continue
                        if h_norm in {"", "n/a"}:
                            continue
                        keep_indices.append(idx)

                    if not keep_indices:
                        continue

                    headers = [headers[i] for i in keep_indices]
                    filtered_rows = []
                    for row in rows:
                        filtered_row = [row[i] if i < len(row) else "" for i in keep_indices]
                        filtered_rows.append(filtered_row)

                    filtered_rows = self._filter_table_rows(headers, filtered_rows)

                    # Try to enforce canonical RACI header order
                    canonical = ["Sl. No.", "Process", "Responsible", "Accountable", "Consulted", "Informed"]
                    header_norms = [h.strip().lower() for h in headers]
                    name_map = {
                        "sl. no.": {"sl. no.", "sl", "sl.", "no", "no."},
                        "process": {"process"},
                        "responsible": {"responsible"},
                        "accountable": {"accountable"},
                        "consulted": {"consulted"},
                        "informed": {"informed"},
                    }
                    index_map = {}
                    for i, h in enumerate(header_norms):
                        for key, aliases in name_map.items():
                            if h in aliases and key not in index_map:
                                index_map[key] = i
                    
                    if len(index_map) >= 4:
                        idxs = [
                            index_map.get("sl. no."),
                            index_map.get("process"),
                            index_map.get("responsible"),
                            index_map.get("accountable"),
                            index_map.get("consulted"),
                            index_map.get("informed"),
                        ]
                        if all(i is not None for i in idxs):
                            headers = canonical
                            filtered_rows = [[row[i] if i < len(row) else "" for i in idxs] for row in filtered_rows]
                    else:
                        # Positional cleanup: trim to 6 columns if needed
                        if len(headers) > 6:
                            headers = headers[:6]
                            filtered_rows = [r[:6] for r in filtered_rows]
                        if len(headers) == 6:
                            headers = canonical

                    if not filtered_rows:
                        continue

                    # Finalize previous table
                    if current_headers and current_rows:
                        table_md = self._format_markdown_table(current_headers, current_rows)
                        page_label = ", ".join(str(p) for p in current_pages)
                        final_tables.append(f"--- Page {page_label} ---\n{table_md}")

                    current_headers = headers
                    current_rows = filtered_rows
                    current_pages = [page_num + 1]

        # Flush last table
        if current_headers and current_rows:
            table_md = self._format_markdown_table(current_headers, current_rows)
            page_label = ", ".join(str(p) for p in current_pages)
            final_tables.append(f"--- Page {page_label} ---\n{table_md}")

        result = "\n\n".join(final_tables)
        return self._clean_table_output(result)

    def extract_raci_table_correct(self, pdf_path: str) -> str:
        """Extract RACI tables with improved detection and fallback logic."""
        import pdfplumber

        final_tables = []
        current_headers = None
        current_rows = []
        current_pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page = self._crop_page_for_tables(page)
                try:
                    tables = page.extract_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                    })
                except Exception:
                    tables = page.extract_tables()
                
                if not tables:
                    continue

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    raw_headers = [str(h).strip() if h else "" for h in table[0]]

                    # Detect RACI table - needs at least 3 of 4 RACI terms
                    header_text = " ".join(raw_headers).lower()
                    raci_keywords = ["responsible", "accountable", "consulted", "informed"]
                    keyword_count = sum(1 for k in raci_keywords if k in header_text)
                    has_raci_header = keyword_count >= 3 or "raci" in header_text

                    rows = []
                    for row in table[1:]:
                        cleaned = []
                        for cell in row:
                            if cell:
                                cleaned.append(str(cell).replace("\n", " ").strip())
                            else:
                                cleaned.append("")
                        while len(cleaned) < len(raw_headers):
                            cleaned.append("")
                        rows.append(cleaned)

                    # Check if first row is a split header
                    if raw_headers and rows and len(rows) > 0:
                        first_row_text = " ".join(x.lower() for x in rows[0])
                        if ("no." in first_row_text or "no" in first_row_text):
                            non_empty_in_first = [x for x in rows[0] if x.strip() and x.strip() not in {"no", "no."}]
                            if len(non_empty_in_first) == 0:
                                raw_headers = [raw_headers[0] + " No."] + raw_headers[1:]
                                rows = rows[1:]

                    # Continuation page handling
                    if not has_raci_header and current_headers and rows:
                        width = len(current_headers)
                        fixed_rows = []
                        for row in rows:
                            if len(row) < width:
                                row = row + [""] * (width - len(row))
                            elif len(row) > width:
                                row = row[:width]
                            fixed_rows.append(row)
                        
                        looks_like_continuation = any(
                            re.match(r"^\d+\s*$", (r[0] or "").strip()) or
                            (len(r) > 1 and (r[1] or "").strip() not in {"", ""})
                            for r in fixed_rows
                        )
                        if looks_like_continuation:
                            fixed_rows = self._filter_table_rows(current_headers, fixed_rows)
                            if fixed_rows:
                                current_rows.extend(fixed_rows)
                                if page_num + 1 not in current_pages:
                                    current_pages.append(page_num + 1)
                        continue

                    if not has_raci_header:
                        continue

                    # Normalize headers/rows
                    headers, rows = self._normalize_table([raw_headers] + rows)
                    if not headers:
                        continue

                    # Keep only relevant columns
                    desired_headers = {"sl.", "sl", "no.", "no", "sl. no.", "process", "responsible", "accountable", "consulted", "informed"}
                    keep_indices = []
                    for idx, h in enumerate(headers):
                        h_norm = h.strip().lower()
                        if h_norm in desired_headers or (h_norm and h_norm not in {"", "n/a"}):
                            keep_indices.append(idx)

                    if not keep_indices:
                        continue

                    headers = [headers[i] for i in keep_indices]
                    filtered_rows = []
                    for row in rows:
                        filtered_row = [row[i] if i < len(row) else "" for i in keep_indices]
                        filtered_rows.append(filtered_row)

                    # Light filtering - preserve rows unless clearly metadata
                    rows_filtered = []
                    for row in filtered_rows:
                        if not any(str(c).strip() and str(c).strip() not in {"N/A", ""} for c in row):
                            continue
                        if self._is_metadata_row(row, headers):
                            continue
                        rows_filtered.append(row)
                    
                    filtered_rows = rows_filtered
                    if not filtered_rows:
                        continue

                    # Flush previous table before starting new one
                    if current_headers and current_rows:
                        table_md = self._format_markdown_table(current_headers, current_rows)
                        page_label = ", ".join(str(p) for p in current_pages)
                        final_tables.append(f"--- Page {page_label} ---\n{table_md}")

                    current_headers = headers
                    current_rows = filtered_rows
                    current_pages = [page_num + 1]

        # Flush last table - CRITICAL: never skip the last accumulated table
        if current_headers and current_rows:
            table_md = self._format_markdown_table(current_headers, current_rows)
            page_label = ", ".join(str(p) for p in current_pages)
            final_tables.append(f"--- Page {page_label} ---\n{table_md}")

        result = "\n\n".join(final_tables)
        return self._clean_table_output(result)

    def extract_sipoc_table_correct(self, pdf_path: str) -> str:
        import pdfplumber

        final_tables = []
        current_headers: Optional[List[str]] = None
        current_rows: List[List[str]] = []
        current_pages: List[int] = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page = self._crop_page_for_tables(page)
                try:
                    tables = page.extract_tables({
                        "vertical_strategy": "lines",
                        "horizontal_strategy": "lines",
                        "intersection_tolerance": 5,
                    })
                except Exception:
                    tables = page.extract_tables()

                for table in tables:
                    if not table or len(table) < 2:
                        continue

                    raw_headers = [str(h).strip() if h else "" for h in table[0]]
                    headers, rows = self._normalize_table([raw_headers] + table[1:])
                    if not headers:
                        continue

                    header_text = " ".join(h.lower() for h in headers)
                    # Check if this looks like a SIPOC table (needs at least 3 of the key terms)
                    sipoc_keywords = ["supplier", "input", "process", "output", "customer"]
                    keyword_count = sum(1 for k in sipoc_keywords if k in header_text)
                    is_sipoc = keyword_count >= 3 or "sipoc" in header_text
                    
                    if not is_sipoc:
                        # Continuation page: same column count, numeric or non-empty first column
                        if current_headers:
                            width = len(current_headers)
                            cont_rows = []
                            # Use ALL rows from table as continuation (do not drop first 2 rows)
                            for row in table:
                                cleaned = []
                                for cell in row:
                                    val = str(cell).replace("\n", " ").strip() if cell else ""
                                    cleaned.append(val if val else "")
                                # Pad/truncate to width
                                if len(cleaned) < width:
                                    cleaned += [""] * (width - len(cleaned))
                                elif len(cleaned) > width:
                                    cleaned = cleaned[:width]
                                # Skip header-like rows
                                joined = " ".join(str(x).lower() for x in cleaned)
                                if keyword_count >= 3 and all(k in joined for k in sipoc_keywords):
                                    continue
                                cont_rows.append(cleaned)

                            cont_rows = self._filter_table_rows(current_headers, cont_rows)
                            if cont_rows:
                                current_rows.extend(cont_rows)
                                if page_num + 1 not in current_pages:
                                    current_pages.append(page_num + 1)
                        continue

                    # Apply filtering but keep more rows for SIPOC (less aggressive)
                    rows_filtered = []
                    for row in rows:
                        # Skip completely empty rows
                        if not any(str(c).strip() and str(c).strip() not in {"N/A", ""} for c in row):
                            continue
                        # Skip rows that are only metadata
                        if self._is_metadata_row(row, headers):
                            continue
                        rows_filtered.append(row)
                    
                    rows = rows_filtered
                    if not rows:
                        continue

                    # Flush previous table
                    if current_headers and current_rows:
                        md = self._format_markdown_table(current_headers, current_rows)
                        page_label = ", ".join(str(p) for p in current_pages)
                        final_tables.append(f"--- Page {page_label} ---\n{md}")

                    current_headers = headers
                    current_rows = rows
                    current_pages = [page_num + 1]

        if current_headers and current_rows:
            md = self._format_markdown_table(current_headers, current_rows)
            page_label = ", ".join(str(p) for p in current_pages)
            final_tables.append(f"--- Page {page_label} ---\n{md}")

        result = "\n\n".join(final_tables)
        return self._clean_table_output(result)

    def extract_change_history_table(self, pdf_filename: str) -> str:
        """Extract Change History table using layout text when table lines aren't detected."""
        keywords = ["change history", "revision history", "revision record", "change log"]
        context = self._get_pdf_pages_with_keywords_layout(pdf_filename, keywords)
        if not context:
            return ""

        lines = [ln.rstrip() for ln in context.splitlines() if ln.strip()]
        if not lines:
            return ""

        header_idx = None
        header_line = ""
        after_marker = False
        header_keywords = {"date", "version", "rev", "revision", "description", "prepared", "approved", "by", "remarks", "author", "changes"}
        for i, line in enumerate(lines):
            lower = line.lower()
            if any(k in lower for k in keywords):
                after_marker = True
                continue
            if after_marker:
                parts = re.split(r"\s{2,}", line.strip())
                parts = [p.strip() for p in parts if p.strip()]
                if len(parts) >= 2:
                    # detect if header-like
                    hits = sum(1 for p in parts if any(k in p.lower() for k in header_keywords))
                    if hits >= 2:
                        header_idx = i
                        header_line = line
                        break
                # Fallback: single-space header line
                parts = line.strip().split()
                hits = sum(1 for p in parts if any(k in p.lower() for k in header_keywords))
                if hits >= 2:
                    header_idx = i
                    header_line = line
                    break

        if header_idx is None:
            return ""

        headers = [p.strip() for p in re.split(r"\s{2,}", header_line) if p.strip()]
        if not headers:
            return ""

        rows = []
        buffer = [""] * len(headers)
        started = False
        for line in lines[header_idx + 1:]:
            lower = line.lower().strip()
            if not lower:
                continue
            if self._is_header_footer_text(lower):
                break
            # stop at next section
            if "document classification" in lower or "document title" in lower:
                break

            parts = re.split(r"\s{2,}", line.strip())
            parts = [p.strip() for p in parts if p.strip()]
            if not parts:
                continue

            # Fill columns; if not enough parts, append to last column
            if len(parts) < len(headers):
                if started:
                    buffer[-1] = (buffer[-1] + " " + " ".join(parts)).strip()
                else:
                    # pad parts
                    parts = parts + ["N/A"] * (len(headers) - len(parts))
                    buffer = [p if p else "N/A" for p in parts]
                    started = True
            else:
                if started and any(c.strip() for c in buffer):
                    rows.append([c.strip() if c.strip() else "N/A" for c in buffer])
                    buffer = [""] * len(headers)
                buffer = [p if p else "N/A" for p in parts[:len(headers)]]
                started = True

        if started and any(c.strip() for c in buffer):
            rows.append([c.strip() if c.strip() else "N/A" for c in buffer])

        if not rows:
            return ""

        table_md = self._format_markdown_table(headers, rows)
        return f"--- Change History ---\n{table_md}"

    def retrieve_docs(self, question: str, k: int = 10) -> List:
        if not self.vector_store:
            self.load_vector_store()
        if not self.vector_store:
            return []
        retriever = self.vector_store.as_retriever(search_kwargs={"k": k})
        source_hint = self._extract_source_hint(question)
        docs = retriever.invoke(question)
        return self._filter_docs_by_source(docs, source_hint)

    def _is_low_quality_answer(self, answer: str) -> bool:
        if not answer:
            return True
        trimmed = answer.strip()
        if len(trimmed) < 20:
            return True
        low = trimmed.lower()
        return low in {"yes.", "no.", "yes", "no", "i don't know", "idk"}



    def answer_question(self, question: str) -> Dict[str, Any]:
        """Retrieves context and generates an answer."""
        if self.llm is None:
            self.llm = self._setup_llm()

        template = """You are an expert Quality SOP assistant. Answer questions accurately using ONLY the context below.

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
1. First, give a clear and direct ANSWER to the question. Be thorough and detailed.
2. Then add a blank line and a "---" separator.
3. Then add a "**References:**" section listing each source with:
   - Document name and page number
   - The exact relevant lines quoted from that page (use quotation marks)

Example format:
[Your detailed answer here]

---
**References:**
- **SOP-Example Document.pdf (page 5):** "exact line from the document that supports the answer"
- **SOP-Another Document.pdf (page 3):** "another exact supporting line"

Rules:
- Use ONLY the context. Do NOT make up information.
- If the answer is not found, respond with: "Not found in the provided SOPs."
- Quote the EXACT lines from the context as references. Do not paraphrase.
- Include ALL relevant source pages, not just one.

Context:
{context}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)

        # Try targeted PDF first when user mentions a specific document
        matched_pdf = self._match_pdf_file(question)
        if matched_pdf:
            context_text = self._get_targeted_pdf_text(matched_pdf, question)
            sources = [matched_pdf]
        else:
            if not self.vector_store:
                self.load_vector_store()
            if not self.vector_store:
                return {"answer": "Vector store not initialized.", "sources": []}
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 10})
            source_hint = self._extract_source_hint(question)
            source_docs = retriever.invoke(question)
            source_docs = self._filter_docs_by_source(source_docs, source_hint)
            if not source_docs:
                return {"answer": "Not found in the provided SOPs.", "sources": []}
            context_text = self.format_docs(source_docs)
            sources = list(set([doc.metadata.get("source", "Unknown") for doc in source_docs]))

        if not context_text:
            return {"answer": "Not found in the provided SOPs.", "sources": []}

        # Truncate context to stay within Groq token limits (~4000 tokens for context)
        if len(context_text) > 20000:
            context_text = context_text[:20000]

        rag_chain = (
            {"context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        try:
            answer = rag_chain.invoke(question)
        except Exception as exc:
            msg = str(exc)
            if "rate limit" in msg.lower() or "rate_limit" in msg.lower() or "429" in msg:
                return {
                    "answer": "Groq rate limit reached. Please wait a few minutes and try again.",
                    "sources": []
                }
            if "ssl connection failed" in msg.lower() or "ssl:" in msg.lower() or "network error" in msg.lower():
                return {
                    "answer": "Groq connection failed because of an SSL/network issue. Please check your internet, proxy, or firewall settings and try again.",
                    "sources": []
                }
            if "timed out" in msg.lower():
                return {
                    "answer": "Groq request timed out. Please try again.",
                    "sources": []
                }
            raise
        if self._is_low_quality_answer(answer):
            answer = "Not found in the provided SOPs."
        
        return {
            "answer": answer,
            "sources": sources
        }

    def extract_only(self, question: str, k: int = 6) -> Dict[str, Any]:
        """Returns verbatim text chunks from PDFs with source + page, no summarization."""
        if not self.vector_store:
            self.load_vector_store()

        if not self.vector_store:
            return {"answer": "Vector store not initialized.", "sources": [], "excerpts": []}

        docs = self.retrieve_docs(question, k=k)
        if not docs:
            return {"answer": "Not found in the provided SOPs.", "sources": [], "excerpts": []}

        excerpts = []
        sources = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            header = f"{source} (page {page + 1})" if page is not None else source
            excerpts.append(f"{header}\n{doc.page_content}")
            sources.append(source)

        return {
            "answer": "",
            "sources": list(set(sources)),
            "excerpts": excerpts,
        }

    def summarize_question(self, question: str, k: int = 6) -> Dict[str, Any]:
        """Returns a concise summary of relevant SOP text with source + page references."""
        if not self.vector_store:
            self.load_vector_store()

        if not self.vector_store:
            return {"summary": "Vector store not initialized.", "sources": []}

        if self.llm is None:
            self.llm = self._setup_llm()

        docs = self.retrieve_docs(question, k=k)
        if not docs:
            return {"summary": "Not found in the provided SOPs.", "sources": []}

        context_text = self.format_docs(docs)

        template = """Summarize the relevant SOP content below in 2-4 concise sentences.
Use ONLY the context. Do not add or infer details not present.
If the context does not answer the question, respond with: "Not found in the provided SOPs."

Context: {context}

Question: {question}

Summary:"""

        prompt = PromptTemplate.from_template(template)
        summarizer = (
            {"context": RunnableLambda(lambda _: context_text), "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        summary = summarizer.invoke(question)

        source_refs = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", None)
            source_refs.append(f"{source} (page {page + 1})" if page is not None else source)

        return {
            "summary": summary,
            "sources": sorted(set(source_refs)),
        }

    @staticmethod
    def _sanitize_mermaid(code: str) -> str:
        """Post-process LLM-generated Mermaid code to fix common syntax issues."""
        if not code:
            return ""

        # Remove code fences
        lines = code.strip().split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            cleaned.append(line)
        code = "\n".join(cleaned).strip()

        # Ensure it starts with flowchart
        if not code.lower().startswith("flowchart"):
            code = "flowchart TD\n" + code

        # Fix node labels: ensure labels with special chars are wrapped in quotes
        # Match patterns like A[label] or A{label} or A(label) and quote the label
        import re as _re

        def _quote_label(m):
            prefix = m.group(1)  # node id
            open_br = m.group(2)  # [ or { or (
            label = m.group(3)    # label text
            close_br = m.group(4) # ] or } or )

            # Remove existing quotes if present, then re-add
            label = label.strip().strip('"').strip("'")
            # Escape any internal double quotes
            label = label.replace('"', "'")
            return f'{prefix}{open_br}"{label}"{close_br}'

        result_lines = []
        for line in code.split("\n"):
            stripped = line.strip()
            # Skip empty lines and the flowchart declaration
            if not stripped or stripped.lower().startswith("flowchart") or stripped.startswith("%%"):
                result_lines.append(line)
                continue

            # Quote labels in node definitions: ID[label], ID{label}, ID(label), ID([label]), ID{{label}}
            # Handle double brackets: ID[["label"]]  and ID{{"label"}}
            line = _re.sub(
                r'(\b\w+)\s*(\[\[|\{\{|\(\[|\[\()(.*?)(\]\]|\}\}|\]\)|\)\])',
                _quote_label, line
            )
            # Handle single brackets: ID[label], ID{label}, ID(label)
            line = _re.sub(
                r'(\b\w+)\s*(\[|\{|\()((?:[^\]\}\)]|\n)*?)(\]|\}|\))',
                _quote_label, line
            )

            result_lines.append(line)

        return "\n".join(result_lines)

    def generate_flowchart(self, question: str) -> Dict[str, Any]:
        """Generates a flowchart image from PDF using fast text-based detection."""
        print(f"[FLOWCHART] Request: {question}")
        matched_pdf = self._match_pdf_file(question)
        
        if matched_pdf:
            print(f"[FLOWCHART] Matched PDF: {matched_pdf}")
            images = self._extract_flowchart_images_fast(matched_pdf, question)
            if images:
                print(f"[FLOWCHART] [OK] Extracted {len(images)} image(s)")
                return {"mermaid": "", "sources": [matched_pdf], "error": "", "images": images}
        else:
            print(f"[FLOWCHART] No direct PDF match. Searching vector store...")
            if not self.vector_store:
                self.load_vector_store()
            if not self.vector_store:
                return {"mermaid": "", "sources": [], "error": "Vector store not initialized."}
            docs = self.retrieve_docs(question, k=5)
            if not docs:
                return {"mermaid": "", "sources": [], "error": "No relevant documents found."}
            matched_pdf = docs[0].metadata.get("source")
            if matched_pdf:
                print(f"[FLOWCHART] Found PDF from vector search: {matched_pdf}")
                images = self._extract_flowchart_images_fast(matched_pdf, question)
                if images:
                    print(f"[FLOWCHART] [OK] Extracted {len(images)} image(s)")
                    return {"mermaid": "", "sources": [matched_pdf], "error": "", "images": images}

        error_msg = f"No flowchart found in {matched_pdf if matched_pdf else 'documents'}. Ensure PDFs have visible flowcharts or flowchart-related text."
        print(f"[FLOWCHART] {error_msg}")
        return {"mermaid": "", "sources": [matched_pdf] if matched_pdf else [], "error": error_msg}

    def generate_table(self, question: str) -> Dict[str, Any]:
        """Generates a markdown table from SOP context using extractors (no hallucinations)."""

        # Try targeted PDF first for comprehensive content
        matched_pdf = self._match_pdf_file(question)
        if matched_pdf:
            question_lower = (question or "").lower()
            pdf_path = str(Path(self.pdf_dir) / matched_pdf)
            
            if "change history" in question_lower:
                ch_table = ""
                try:
                    ch_table = self.extract_change_history_table(matched_pdf)
                except Exception:
                    ch_table = ""
                if ch_table:
                    return {"table": self._clean_table_output(ch_table), "sources": [matched_pdf], "error": ""}
            
            if "raci" in question_lower:
                raci_tables = ""
                try:
                    raci_tables = self.extract_raci_table_correct(pdf_path)
                except Exception:
                    raci_tables = ""
                if raci_tables:
                    return {"table": self._clean_table_output(raci_tables), "sources": [matched_pdf], "error": ""}
                
                # Fallback: try generic extraction with RACI keywords
                fallback_raci = ""
                try:
                    fallback_raci = self.extract_any_table_with_keywords(
                        pdf_path,
                        ["responsible", "accountable", "consulted", "informed"]
                    )
                except Exception:
                    fallback_raci = ""
                if fallback_raci:
                    return {"table": self._clean_table_output(fallback_raci), "sources": [matched_pdf], "error": ""}
            
            if "sipoc" in question_lower:
                sipoc_tables = ""
                try:
                    sipoc_tables = self.extract_sipoc_table_correct(pdf_path)
                except Exception:
                    sipoc_tables = ""
                if sipoc_tables:
                    return {"table": self._clean_table_output(sipoc_tables), "sources": [matched_pdf], "error": ""}
                
                # Fallback: try generic extraction with SIPOC keywords
                fallback_sipoc = ""
                try:
                    fallback_sipoc = self.extract_any_table_with_keywords(
                        pdf_path, 
                        ["supplier", "input", "process", "output", "customer"]
                    )
                except Exception:
                    fallback_sipoc = ""
                if fallback_sipoc:
                    return {"table": self._clean_table_output(fallback_sipoc), "sources": [matched_pdf], "error": ""}
            
            paddle_tables = ""
            try:
                paddle_tables = self.extract_tables_paddleocr(pdf_path)
            except Exception:
                paddle_tables = ""
            if paddle_tables:
                return {"table": self._clean_table_output(paddle_tables), "sources": [matched_pdf], "error": ""}
        else:
            if not self.vector_store:
                self.load_vector_store()
            if not self.vector_store:
                return {"table": "", "sources": [], "error": "Vector store not initialized."}
            docs = self.retrieve_docs(question, k=10)
            if not docs:
                return {"table": "", "sources": [], "error": "No relevant documents found."}
            matched_pdf = docs[0].metadata.get("source")
            if not matched_pdf:
                return {"table": "", "sources": [], "error": "No relevant document found."}
            
            pdf_path = str(Path(self.pdf_dir) / matched_pdf)
            question_lower = (question or "").lower()
            
            if "change history" in question_lower:
                ch_table = ""
                try:
                    ch_table = self.extract_change_history_table(matched_pdf)
                except Exception:
                    ch_table = ""
                if ch_table:
                    return {"table": self._clean_table_output(ch_table), "sources": [matched_pdf], "error": ""}
            
            if "sipoc" in question_lower:
                sipoc_tables = ""
                try:
                    sipoc_tables = self.extract_sipoc_table_correct(pdf_path)
                except Exception:
                    sipoc_tables = ""
                if sipoc_tables:
                    return {"table": self._clean_table_output(sipoc_tables), "sources": [matched_pdf], "error": ""}
                
                # Fallback: try generic extraction with SIPOC keywords
                fallback_sipoc = ""
                try:
                    fallback_sipoc = self.extract_any_table_with_keywords(
                        pdf_path,
                        ["supplier", "input", "process", "output", "customer"]
                    )
                except Exception:
                    fallback_sipoc = ""
                if fallback_sipoc:
                    return {"table": self._clean_table_output(fallback_sipoc), "sources": [matched_pdf], "error": ""}
            
            paddle_tables = ""
            try:
                paddle_tables = self.extract_tables_paddleocr(pdf_path)
            except Exception:
                paddle_tables = ""
            if paddle_tables:
                return {"table": self._clean_table_output(paddle_tables), "sources": [matched_pdf], "error": ""}

        # Camelot fallback (no LLM) if pdfplumber/raci failed
        camelot_tables_text = ""
        try:
            import camelot  # type: ignore

            pdf_path = str(Path(self.pdf_dir) / matched_pdf) if matched_pdf else ""
            if pdf_path and os.path.exists(pdf_path):
                tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
                if tables.n == 0:
                    tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
                if tables:
                    table_texts = []
                    for i, t in enumerate(tables, start=1):
                        df = t.df
                        if "sipoc" in (question or "").lower():
                            header_row = [str(x).strip().lower() for x in df.iloc[0].tolist()]
                            header_text = " ".join(header_row)
                            keyword_count = sum(1 for k in ["supplier", "input", "process", "output", "customer"] if k in header_text)
                            is_sipoc = keyword_count >= 3 or "sipoc" in header_text
                            if not is_sipoc:
                                continue
                        table_texts.append(f"--- Camelot Table {i} ---\n{df.to_markdown(index=False)}")
                    camelot_tables_text = "\n\n".join(table_texts)
        except Exception:
            camelot_tables_text = ""

        if camelot_tables_text:
            return {"table": self._clean_table_output(camelot_tables_text), "sources": [matched_pdf], "error": ""}

        # Last resort: if change history requested, scan PDFs for keyword and retry
        if "change history" in (question or "").lower():
            fallback_pdf = self._find_pdf_by_keywords(["change history", "revision history", "change log"])
            if fallback_pdf:
                ch_table = ""
                try:
                    ch_table = self.extract_change_history_table(fallback_pdf)
                except Exception:
                    ch_table = ""
                if ch_table:
                    return {"table": self._clean_table_output(ch_table), "sources": [fallback_pdf], "error": ""}

        return {"table": "", "sources": [matched_pdf] if matched_pdf else [], "error": "No tables extracted from the document."}

    def generate_text(self, prompt: str) -> str:
        """Direct LLM call for auxiliary tasks (e.g., flowchart reconstruction)."""
        if self.llm is None:
            self.llm = self._setup_llm()
        if hasattr(self.llm, "invoke"):
            return self.llm.invoke(prompt)
        return self.llm(prompt)
