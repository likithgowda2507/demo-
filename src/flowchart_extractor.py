import os
import re
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import importlib


class FlowchartExtractor:
    """
    Image/flowchart extraction pipeline using PyMuPDF.
    - Searches matched PDF first, then falls back to all PDFs.
    - Uses LLM to generate a Mermaid diagram when no image is found.
    - Produces tight, precisely-cropped flowchart images.
    """

    def __init__(self, pdf_dir: str, images_dir: str, llm=None):
        self.pdf_dir = pdf_dir
        self.images_dir = images_dir
        self.llm = llm
        os.makedirs(self.images_dir, exist_ok=True)

    # ─────────────────────────── Public API ───────────────────────────

    def extract_flowcharts(
        self,
        question: str,
        matched_pdf: Optional[str] = None,
        max_images: int = 2,
    ) -> Dict[str, Any]:
        try:
            fitz = importlib.import_module("fitz")
        except Exception:
            return {
                "images": [],
                "mermaid": "",
                "sources": [],
                "error": "PyMuPDF is not installed. Please install pymupdf.",
            }

        # ── Step 1: determine which PDFs to search ──────────────────
        pdf_name = matched_pdf or self._match_pdf_file(question)
        if pdf_name:
            pdf_candidates = [pdf_name]
        else:
            # Search every PDF in directory — broadest fallback
            root = Path(self.pdf_dir)
            pdf_candidates = [p.name for p in sorted(root.glob("*.pdf"))] if root.exists() else []

        if not pdf_candidates:
            return {
                "images": [],
                "mermaid": self._llm_mermaid(question, context=""),
                "sources": [],
                "error": "No PDF documents found.",
            }

        # ── Step 2: extract from each candidate PDF ──────────────────
        keywords = [
            "flowchart", "flow chart", "process flow", "process flow chart",
            "overall process flow", "workflow", "diagram", "process diagram",
            "swimlane", "activity diagram",
        ]
        query_terms = self._query_terms(question)

        if not matched_pdf and len(pdf_candidates) > 1:
            pdf_candidates = self._rank_pdf_candidates(question, pdf_candidates, query_terms)

        images: List[bytes] = []
        sources: List[str] = []
        context_text: str = ""

        for pdf_name_candidate in pdf_candidates:
            pdf_path = str(Path(self.pdf_dir) / pdf_name_candidate)
            if not os.path.exists(pdf_path):
                continue

            result_images, result_sources, result_text = self._extract_from_pdf(
                fitz, pdf_path, pdf_name_candidate, question, keywords, query_terms, max_images - len(images)
            )
            images.extend(result_images)
            sources.extend(result_sources)
            if result_text:
                context_text += result_text + "\n\n"

            if len(images) >= max_images:
                break

        # ── Step 3: LLM Mermaid fallback if no images ───────────────
        mermaid_code = ""
        if not images and self.llm is not None:
            mermaid_code = self._llm_mermaid(question, context_text)

        if not images and not mermaid_code:
            return {
                "images": [],
                "mermaid": "",
                "sources": sources or list({matched_pdf} if matched_pdf else []),
                "error": "No flowchart image found in the document.",
            }

        return {"images": images, "mermaid": mermaid_code, "sources": sources, "error": ""}

    # ─────────────────────────── PDF Extraction ───────────────────────────

    def _extract_from_pdf(
        self,
        fitz: Any,
        pdf_path: str,
        pdf_name: str,
        question: str,
        keywords: List[str],
        query_terms: List[str],
        max_images: int,
    ) -> Tuple[List[bytes], List[str], str]:
        images: List[bytes] = []
        sources: List[str] = []
        context_text = ""
        selected_pages = set()

        doc = fitz.open(pdf_path)
        try:
            ranked_pages = self._rank_flowchart_pages(doc, question, keywords, query_terms)

            # ── Primary pass: render cropped page ──
            for page_idx, score in ranked_pages:
                page = doc[page_idx]
                page_text = page.get_text("text") or ""
                if self._is_skip_page(page_text, page_idx):
                    continue
                if not self._is_likely_flowchart_page(fitz, page, page_text):
                    continue

                img_bytes = self._render_best_crop(fitz, page, keywords, query_terms)
                if img_bytes is None:
                    continue

                out_path = Path(self.images_dir) / f"{Path(pdf_name).stem}_page_{page_idx + 1}.png"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

                images.append(img_bytes)
                sources.append(f"{pdf_name} (page {page_idx + 1})")
                context_text += f"\n[Page {page_idx + 1} of {pdf_name}]\n{page_text[:1500]}"
                selected_pages.add(page_idx)

                # If a flowchart continues on the following page, include it.
                if len(images) < max_images:
                    next_idx = page_idx + 1
                    if next_idx < len(doc) and next_idx not in selected_pages:
                        next_page = doc[next_idx]
                        next_text = next_page.get_text("text") or ""
                        if not self._is_skip_page(next_text, next_idx):
                            if self._is_likely_flowchart_page(fitz, next_page, next_text):
                                next_img_bytes = self._render_best_crop(fitz, next_page, keywords, query_terms)
                                if next_img_bytes is not None:
                                    out_path = Path(self.images_dir) / f"{Path(pdf_name).stem}_page_{next_idx + 1}.png"
                                    with open(out_path, "wb") as f:
                                        f.write(next_img_bytes)
                                    images.append(next_img_bytes)
                                    sources.append(f"{pdf_name} (page {next_idx + 1})")
                                    context_text += f"\n[Page {next_idx + 1} of {pdf_name}]\n{next_text[:1500]}"
                                    selected_pages.add(next_idx)

                if len(images) >= max_images:
                    break

            # ── Fallback pass: embedded images ──
            if not images:
                for page_idx, _ in ranked_pages:
                    page = doc[page_idx]
                    page_text = page.get_text("text") or ""
                    if self._is_skip_page(page_text, page_idx):
                        continue
                    if not self._is_likely_flowchart_page(fitz, page, page_text):
                        continue
                    if not (self._has_diagram_signal(fitz, page) or self._has_strong_flow_heading(page_text)):
                        continue

                    for img_no, img in enumerate(page.get_images(full=True)):
                        xref = img[0]
                        base = doc.extract_image(xref)
                        data = base.get("image", b"")
                        if not data:
                            continue
                        w, h = base.get("width", 0), base.get("height", 0)
                        if w < 300 or h < 180:
                            continue

                        out_path = (
                            Path(self.images_dir)
                            / f"{Path(pdf_name).stem}_page_{page_idx + 1}_img_{img_no + 1}.png"
                        )
                        with open(out_path, "wb") as f:
                            f.write(data)

                        images.append(data)
                        sources.append(f"{pdf_name} (page {page_idx + 1})")
                        context_text += f"\n[Page {page_idx + 1} of {pdf_name}]\n{page_text[:1500]}"

                        if len(images) >= max_images:
                            break
                    if len(images) >= max_images:
                        break
        finally:
            doc.close()

        return images, sources, context_text

    # ─────────────────────────── Smart Crop ───────────────────────────

    def _render_best_crop(
        self,
        fitz: Any,
        page: Any,
        keywords: List[str],
        query_terms: List[str],
    ) -> Optional[bytes]:
        """
        Render the most relevant clipped area of the page at high DPI.
        Uses drawing boxes + image rects + keyword text boxes to compute
        a tight bounding rect. Falls back to full page minus margins.
        """
        page_rect = page.rect
        boilerplate_bottom = self._boilerplate_header_bottom(page)
        boilerplate_top_footer = self._boilerplate_footer_top(page)

        # Collect drawing bounding boxes (vector shapes)
        draw_boxes = []
        try:
            for d in page.get_drawings():
                r = d.get("rect")
                if r:
                    fr = fitz.Rect(r)
                    if fr.width > 30 and fr.height > 15:
                        draw_boxes.append(fr)
        except Exception:
            pass

        # Collect embedded image bounding boxes
        image_boxes = []
        try:
            imgs = page.get_images(full=True)
            for img in imgs:
                rects = page.get_image_rects(img[0])
                for r in rects:
                    fr = fitz.Rect(r)
                    if fr.width > 100 and fr.height > 60:
                        image_boxes.append(fr)
        except Exception:
            pass

        # Collect keyword-matching text blocks
        text_boxes = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            low = (text or "").lower()
            if self._is_boilerplate_text(low):
                continue
            if any(k in low for k in keywords) or (query_terms and any(t in low for t in query_terms)):
                text_boxes.append(fitz.Rect(x0, y0, x1, y1))

        all_boxes = draw_boxes + image_boxes
        if not all_boxes:
            # Only text hints — expand downward from the heading
            if text_boxes:
                heading = min(text_boxes, key=lambda r: r.y0)
                clip = fitz.Rect(
                    page_rect.x0 + 8,
                    max(page_rect.y0, heading.y0 - 10, boilerplate_bottom),
                    page_rect.x1 - 8,
                    min(page_rect.y1 - 8, boilerplate_top_footer),
                )
                clip &= page_rect
                if clip.width > page_rect.width * 0.2 and clip.height > page_rect.height * 0.2:
                    return self._pixmap_bytes(fitz, page, clip)
            return None

        # Find heading anchor and prefer boxes below it
        if text_boxes:
            heading = min(text_boxes, key=lambda r: r.y0)
            below = [r for r in all_boxes if r.y0 >= heading.y0 - 20]
            if below:
                all_boxes = below

        # Union all candidate boxes
        union = all_boxes[0]
        for r in all_boxes[1:]:
            union |= r

        if text_boxes:
            heading = min(text_boxes, key=lambda r: r.y0)
            union |= heading

        # Tight crop with small padding
        pad = 18
        clip = fitz.Rect(
            union.x0 - pad,
            union.y0 - pad,
            union.x1 + pad,
            union.y1 + pad,
        )
        clip &= page_rect
        if boilerplate_bottom > clip.y0:
            clip = fitz.Rect(clip.x0, boilerplate_bottom, clip.x1, clip.y1)
            clip &= page_rect
        if boilerplate_top_footer < clip.y1:
            clip = fitz.Rect(clip.x0, clip.y0, clip.x1, boilerplate_top_footer)
            clip &= page_rect

        # If clip is tiny, use full page
        if clip.width < page_rect.width * 0.12 or clip.height < page_rect.height * 0.12:
            clip = fitz.Rect(
                page_rect.x0 + 8,
                max(page_rect.y0 + 8, boilerplate_bottom),
                page_rect.x1 - 8,
                min(page_rect.y1 - 8, boilerplate_top_footer),
            )

        return self._pixmap_bytes(fitz, page, clip)

    def _pixmap_bytes(self, fitz: Any, page: Any, clip: Any) -> Optional[bytes]:
        """Render a clip at high DPI (3x), convert to PNG bytes."""
        try:
            mat = fitz.Matrix(3.0, 3.0)
            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            if pix.width < 200 or pix.height < 150:
                # too small — try full page at 2.5x
                pix = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), alpha=False)
            return pix.tobytes("png")
        except Exception:
            return None

    # ─────────────────────────── Page Ranking ───────────────────────────

    def _rank_flowchart_pages(
        self, doc: Any, question: str, keywords: List[str], query_terms: List[str]
    ) -> List[Tuple[int, float]]:
        import importlib
        fitz = importlib.import_module("fitz")

        page_texts = [(doc[i].get_text("text") or "").lower() for i in range(len(doc))]
        page_contexts = [self._neighbor_context(page_texts, i) for i in range(len(doc))]

        term_doc_freq: Dict[str, int] = {}
        for term in query_terms:
            term_doc_freq[term] = sum(1 for ctx in page_contexts if term in ctx)

        focus_phrase = self._focus_phrase(question)
        
        ranked: List[Tuple[int, float]] = []
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            text = page_texts[page_idx]
            context_text = page_contexts[page_idx]
            
            if self._is_skip_page(text, page_idx):
                continue

            keyword_hits = sum(1 for k in keywords if k in text)
            
            # Query score uses neighboring context to catch titles at page boundaries.
            query_present = [t for t in query_terms if t in context_text]
            query_hits = len(query_present)
            query_score = 0.0
            for t in query_present:
                df = term_doc_freq.get(t, 1)
                query_score += 1.0 + (len(doc) / max(df, 1)) * 0.35

            if query_terms:
                coverage = query_hits / max(len(query_terms), 1)
                if coverage >= 0.45:
                    query_score += 18
                elif coverage >= 0.25:
                    query_score += 8
                elif coverage == 0:
                    query_score -= 12

            if focus_phrase and focus_phrase in context_text:
                query_score += 14

            diagram_signal = self._diagram_signal(fitz, page)

            # Weight visual features and explicit keywords heavily
            score = (keyword_hits * 6) + query_score + diagram_signal

            # Heavily penalize RACI tables / Matrices. A RACI matrix table grid
            # is drawn with lines, which artificially balloons the diagram_signal.
            if re.search(r"\braci\b", text):
                score -= 50
            elif "responsible" in text and "accountable" in text and "consulted" in text:
                score -= 50
            elif sum(1 for w in ["matrix", "table", "role", "responsibility"] if w in text) >= 3:
                score -= 20

            # Penalize pages lacking visual content
            if diagram_signal < 4:
                score -= 10

            # Strongly down-rank prose-heavy pages to avoid text-image outputs.
            if self._looks_like_text_page(text) and diagram_signal < 22:
                score -= 30

            if not self._is_likely_flowchart_page(fitz, page, text):
                score -= 45
                
            if self._has_strong_flow_heading(text):
                score += 15

            if score <= 0:
                continue

            ranked.append((page_idx, score))

        # ── Title-page lookahead: "Title on page N, flowchart on page N+1" fix ──
        # For each page with high query relevance but low diagram signal (a title heading page),
        # propagate 75% of its score to the next page that actually has the diagram.
        raw_scores: Dict[int, float] = {p: s for p, s in ranked}
        for page_idx, score in list(raw_scores.items()):
            page = doc[page_idx]
            ds = self._diagram_signal(fitz, page)
            # Identify title pages: good relevance but weak visual diagram signal
            if ds < 12 and score > 15:
                for offset in [1, 2]:
                    next_idx = page_idx + offset
                    if next_idx >= len(doc):
                        break
                    next_page = doc[next_idx]
                    next_text = page_texts[next_idx]
                    next_ds = self._diagram_signal(fitz, next_page)
                    if self._is_skip_page(next_text, next_idx):
                        break
                    # If next page has a real diagram, boost it with the title page's relevance
                    if next_ds >= 10:
                        carry = score * 0.75
                        raw_scores[next_idx] = raw_scores.get(next_idx, 0.0) + carry
                        break  # only propagate to the first visual page

        ranked = [(p, s) for p, s in raw_scores.items() if s > 0]
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:10]

    def _neighbor_context(self, page_texts: List[str], page_idx: int) -> str:
        prev_tail = page_texts[page_idx - 1][-1200:] if page_idx > 0 else ""
        cur = page_texts[page_idx]
        next_head = page_texts[page_idx + 1][:1200] if page_idx + 1 < len(page_texts) else ""
        return f"{prev_tail}\n{cur}\n{next_head}"

    def _focus_phrase(self, question: str) -> str:
        q = self._normalize_text(question)
        if not q:
            return ""
        m = re.search(r"\b(?:for|of|about|on)\b\s+(.+)$", q)
        phrase = m.group(1).strip() if m else q
        tokens = [
            t
            for t in phrase.split()
            if t not in {
                "show", "give", "me", "the", "a", "an", "flow", "chart", "flowchart",
                "process", "diagram", "workflow", "sop", "please", "overall",
            }
        ]
        if len(tokens) < 2:
            return ""
        return " ".join(tokens[:7])

    def _looks_like_text_page(self, text: str) -> bool:
        low = (text or "").lower()
        words = low.split()
        if len(words) < 120:
            return False

        flow_terms = [
            "flowchart", "flow chart", "process flow", "workflow", "decision", "start", "end", "step",
        ]
        flow_hits = sum(1 for t in flow_terms if t in low)
        paragraph_markers = sum(low.count(m) for m in ["\n-", "\n*", "\n•", " shall ", " should ", " procedure "])
        dense_lines = sum(1 for ln in low.splitlines() if len(ln.split()) >= 10)

        # Large prose content with weak flow markers is likely a text section page.
        return flow_hits <= 1 and (paragraph_markers >= 2 or dense_lines >= 6)

    def _is_likely_flowchart_page(self, fitz: Any, page: Any, text: str) -> bool:
        low = (text or "").lower()
        words = low.split()
        word_count = len(words)
        dense_lines = sum(1 for ln in low.splitlines() if len(ln.split()) >= 10)
        flow_term_hits = sum(1 for t in ["flowchart", "flow chart", "process flow", "workflow", "start", "end", "decision"] if t in low)
        diagram_signal = self._diagram_signal(fitz, page)

        # Strong visual diagrams should pass even if title text is sparse.
        if diagram_signal >= 24:
            return True

        # Typical flowchart pages: visual signal + not prose-heavy.
        if diagram_signal >= 12 and dense_lines <= 6 and word_count <= 220:
            return True

        # Explicit flow heading with at least some diagram signal.
        if self._has_strong_flow_heading(low) and diagram_signal >= 6:
            return True

        # Reject prose-heavy pages that may include incidental boxes/lines.
        if self._looks_like_text_page(low) and diagram_signal < 24:
            return False
        if word_count > 240 and dense_lines >= 8 and flow_term_hits <= 1:
            return False

        return flow_term_hits >= 2 and diagram_signal >= 8

    def _rank_pdf_candidates(self, question: str, pdf_names: List[str], query_terms: List[str]) -> List[str]:
        """Sort PDFs so the most question-relevant documents are searched first."""
        try:
            fitz = importlib.import_module("fitz")
        except Exception:
            return pdf_names

        focus_phrase = self._focus_phrase(question)
        q_tokens = set(self._normalize_text(question).split())
        filename_stop = {"sop", "ut", "v", "issue", "process", "flow", "chart", "training"}

        scored: List[Tuple[str, float]] = []
        for pdf_name in pdf_names:
            pdf_path = Path(self.pdf_dir) / pdf_name
            if not pdf_path.exists():
                continue

            score = 0.0

            # File-name overlap provides a cheap initial hint.
            stem_tokens = set(self._normalize_text(Path(pdf_name).stem).split()) - filename_stop
            score += len(stem_tokens & q_tokens) * 4.0

            doc = None
            try:
                doc = fitz.open(str(pdf_path))
                text_chunks: List[str] = []
                for i in range(len(doc)):
                    text_chunks.append((doc[i].get_text("text") or "").lower())
                preview = "\n".join(text_chunks)

                if focus_phrase and focus_phrase in preview:
                    score += 24.0

                for term in query_terms:
                    if term in preview:
                        score += 3.0

                if "flow" in preview and "chart" in preview:
                    score += 4.0
            except Exception:
                pass
            finally:
                if doc is not None:
                    doc.close()

            scored.append((pdf_name, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in scored] if scored else pdf_names

    # ─────────────────────────── LLM Mermaid Fallback ───────────────────────────

    def _llm_mermaid(self, question: str, context: str) -> str:
        """Ask the LLM to generate a Mermaid flowchart from text context."""
        if self.llm is None:
            return ""
        try:
            ctx = context.strip()[:4000] if context else ""
            prompt = (
                "You are an expert at creating Mermaid.js flowchart diagrams from SOP documents.\n\n"
                "Based on the user question and SOP context below, generate a valid Mermaid flowchart diagram.\n"
                "Rules:\n"
                "- Use `graph TD` (top-down) layout.\n"
                "- Include clear, labeled nodes and arrows.\n"
                "- Only describe steps/processes found in the context.\n"
                "- If context is empty, generate a reasonable process diagram based on the question.\n"
                "- Output ONLY the raw Mermaid code, no markdown fences, no explanation.\n\n"
                f"User Question: {question}\n\n"
                f"SOP Context:\n{ctx if ctx else 'No context available.'}\n\n"
                "Mermaid Code:"
            )
            result = self.llm.invoke(prompt)
            code = str(result).strip()
            # Strip markdown fences if present
            code = re.sub(r"^```(?:mermaid)?\s*", "", code, flags=re.IGNORECASE)
            code = re.sub(r"\s*```$", "", code)
            code = code.strip()
            if "graph" not in code.lower() and "flowchart" not in code.lower():
                return ""
            return code
        except Exception:
            return ""

    # ─────────────────────────── Helpers ───────────────────────────

    def _is_skip_page(self, text: str, page_idx: int) -> bool:
        low = (text or "").lower()
        if not low.strip():
            return True

        # Title / Cover pages often have very few words or distinct markers
        word_count = len(low.split())
        if page_idx == 0 and (word_count < 80 or "document title" in low or "document no" in low):
            return True

        # Table of contents
        if "table of contents" in low or re.search(r"\btable of content\b", low):
            return True
        if re.search(r"^\s*contents\b", low, flags=re.MULTILINE):
            return True

        # Revision / Change history (these are typically text/table data, not visual flowcharts)
        if "revision history" in low or "change history" in low or "document history" in low:
            # ONLY skip if it doesn't also prominently mention flowchart words
            if "flowchart" not in low and "diagram" not in low:
                return True

        dotted = len(re.findall(r"\.{3,}", low))
        lines = [ln.strip() for ln in low.splitlines() if ln.strip()]
        line_num_endings = sum(1 for ln in lines if re.search(r"\b\d{1,3}\s*$", ln))
        if dotted >= 4 and line_num_endings >= 4:
            return True
        return False

    def _diagram_signal(self, fitz: Any, page: Any) -> float:
        draw_count = 0
        try:
            for d in page.get_drawings():
                r = d.get("rect")
                if r:
                    fr = fitz.Rect(r)
                    if fr.width > 20 and fr.height > 20:
                        draw_count += 1
        except Exception:
            pass

        img_score = 0
        try:
            for img in page.get_images(full=True):
                rects = page.get_image_rects(img[0])
                for r in rects:
                    fr = fitz.Rect(r)
                    if fr.width > 250 and fr.height > 150:
                        img_score += 15
                    elif fr.width > 100 and fr.height > 60:
                        img_score += 5
        except Exception:
            pass

        return min(draw_count, 60) * 0.5 + img_score

    def _has_diagram_signal(self, fitz: Any, page: Any) -> bool:
        return self._diagram_signal(fitz, page) >= 6

    def _has_strong_flow_heading(self, text: str) -> bool:
        low = (text or "").lower()
        patterns = [
            "overall process flow", "process flow chart", "flow chart",
            "workflow", "process diagram", "activity diagram", "swimlane",
        ]
        return any(p in low for p in patterns)

    def _is_boilerplate_text(self, text: str) -> bool:
        low = (text or "").lower()
        keys = [
            "document title", "document no", "effective date", "next review",
            "version", "issue", "document classification", "document status",
            "document template", "confidential", "classified", "page ",
        ]
        return any(k in low for k in keys)

    def _boilerplate_header_bottom(self, page: Any) -> float:
        """Return y-coordinate below detected SOP header boilerplate."""
        page_rect = page.rect
        cutoff = page_rect.y0
        max_scan_y = page_rect.y0 + (page_rect.height * 0.35)

        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            if y0 > max_scan_y:
                continue
            low = (text or "").lower()
            if self._is_boilerplate_text(low):
                cutoff = max(cutoff, float(y1) + 10)

        return min(cutoff, page_rect.y0 + page_rect.height * 0.5)

    def _boilerplate_footer_top(self, page: Any) -> float:
        """Return y-coordinate above detected footer boilerplate."""
        page_rect = page.rect
        cutoff = page_rect.y1
        min_scan_y = page_rect.y0 + (page_rect.height * 0.65)

        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            x0, y0, x1, y1, text = block[:5]
            if y1 < min_scan_y:
                continue
            low = (text or "").lower()
            if self._is_boilerplate_text(low):
                cutoff = min(cutoff, float(y0) - 8)
                continue
            if re.search(r"\bpage\s*\d+(?:\s*of\s*\d+)?\b", low):
                cutoff = min(cutoff, float(y0) - 8)
                continue
            if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", low):
                cutoff = min(cutoff, float(y0) - 8)

        return max(cutoff, page_rect.y0 + page_rect.height * 0.45)

    def _query_terms(self, question: str) -> List[str]:
        q = self._normalize_text(question)
        stop = {
            "show", "give", "me", "the", "a", "an", "for", "of", "to", "from", "in",
            "on", "flow", "chart", "flowchart", "process", "diagram", "workflow",
            "sop", "please", "what", "how", "does", "is", "are", "get",
        }
        return [w for w in q.split() if len(w) >= 3 and w not in stop]

    def _normalize_text(self, text: str) -> str:
        return " ".join(re.sub(r"[^a-zA-Z0-9\s]", " ", (text or "").lower()).split())

    def _match_pdf_file(self, question: str) -> Optional[str]:
        q = self._normalize_text(question)
        if not q:
            return None
        root = Path(self.pdf_dir)
        if not root.exists():
            return None

        best = None
        best_score = 0
        for pdf in root.glob("*.pdf"):
            stem = self._normalize_text(pdf.stem)
            if not stem:
                continue
            if stem in q and len(stem) > best_score:
                best = pdf.name
                best_score = len(stem)

            words = set(stem.split()) - {"sop", "ut", "of", "and", "the", "in", "for", "a", "an", "to"}
            overlap = words & set(q.split())
            if len(words) > 0 and len(overlap) >= max(2, int(len(words) * 0.5)) and len(overlap) > best_score:
                best = pdf.name
                best_score = len(overlap)

        return best
