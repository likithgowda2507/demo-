# pyre-ignore-all-errors
import os
import re
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TableExtractor:
    """
    Deterministic table extractor based on pdfplumber.

    Design goals:
    - Extract clean tables directly from PDFs (no LLM table generation).
    - Stitch continuation tables across pages ("PDF stitch").
    - Normalize common SOP table types (RACI, SIPOC, change history).
    """

    TYPE_KEYWORDS: Dict[str, List[str]] = {
        "raci": ["responsible", "accountable", "consulted", "informed", "raci", "activity", "process"],
        "sipoc": ["supplier", "input", "process", "output", "customer", "sipoc", "control"],
        "generic": [],
    }

    RIVAL_KEYWORDS: Dict[str, List[str]] = {
        "raci": ["supplier", "input", "output", "customer", "sipoc"],
        "sipoc": ["responsible", "accountable", "consulted", "informed", "raci"],
        "generic": [],
    }

    def __init__(self, pdf_dir: str, llm=None):
        self.pdf_dir = pdf_dir
        self.llm = llm
        self.catalog_path = str(Path(self.pdf_dir) / "table_catalog.json")
        self._table_catalog_cache: Optional[Dict[str, Any]] = None
        self.meta_patterns = [
            r"^document title:?", r"^document no:?", r"^document classification:?",
            r"^document status:?", r"^document template:?", r"^effective date:?",
            r"^next review:?", r"^confidential\s*$", r"^classified\s*$",
            r"^cannot be shared\s*$", r"\bnda\b", r"^page\s*\d+\s*(of\s*\d+)?\s*$",
            r"^page\s*\d+\s*$",
        ]

    # ----------------------------- Public API -----------------------------

    def extract_table(
        self,
        question: str,
        matched_pdf: Optional[str] = None,
        forced_table_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        table_type = forced_table_type or self._table_type(question)
        pdf_name = matched_pdf or self._match_pdf_file(question)
        if not pdf_name:
            pdf_name = self._find_pdf_by_table_type(table_type)
        if not pdf_name:
            return {"table": "", "sources": [], "error": "No relevant document found."}

        pdf_path = str(Path(self.pdf_dir) / pdf_name)
        if not os.path.exists(pdf_path):
            return {"table": "", "sources": [pdf_name], "error": "Document not found."}

        tables, pages = self._extract_pdfplumber_tables(pdf_path, table_type, question)
        if not tables:
            tables, pages = self._extract_camelot_tables(pdf_path, table_type, question)
        if not tables and table_type in {"raci", "sipoc"}:
            tables, pages = self._extract_type_fallback_from_all_tables(pdf_path, table_type)

        if not tables:
            return {"table": "", "sources": [pdf_name], "error": "No table extracted from the document."}

        merged_h, merged_r, merged_pages = self._merge_multipage_tables(tables, pages)
        final_h, final_r = self._postprocess_table_for_type(merged_h, merged_r, table_type)

        if not self._is_viable_typed_table(final_h, final_r, table_type):
            return {
                "table": "",
                "sources": [f"{pdf_name} (page {p})" for p in merged_pages] if merged_pages else [pdf_name],
                "error": "No valid table found for the requested type.",
            }

        md = self._to_markdown(final_h, final_r)
        found_pages = merged_pages or pages
        page_block = f"--- Page {', '.join(str(p) for p in found_pages)} ---\n"
        page_labels = [f"{pdf_name} (page {p})" for p in found_pages]
        return {"table": page_block + md, "sources": page_labels, "error": "", "llm_cleaned": False}

    def build_table_catalog(self, force: bool = False) -> Dict[str, Any]:
        """
        Scan all PDFs and cache where RACI/SIPOC tables exist.
        """
        return self._load_or_build_table_catalog(force=force)

    def _extract_type_fallback_from_all_tables(
        self,
        pdf_path: str,
        table_type: str,
    ) -> Tuple[List[Tuple[List[str], List[List[str]]]], List[int]]:
        """
        Broad fallback scan across all raw tables for typed requests.
        Useful for PDFs where strict header scoring misses non-standard layouts.
        """
        try:
            import pdfplumber
        except Exception:
            return [], []

        candidates: List[Tuple[float, int, List[str], List[List[str]]]] = []
        settings_candidates = [
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            },
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "intersection_tolerance": 3,
            },
            None,
        ]

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                for settings in settings_candidates:
                    try:
                        raw_tables = page.extract_tables(settings) if settings else (page.extract_tables() or [])
                    except Exception:
                        raw_tables = []

                    for raw in raw_tables or []:
                        h, r = self._normalize_raw_table(raw)
                        if not h or not r:
                            continue
                        nh, nr = self._postprocess_table_for_type(h, r, table_type)
                        if not nh or not nr:
                            continue
                        if not self._is_viable_typed_table(nh, nr, table_type):
                            continue

                        joined = " ".join(nh + [" ".join(x) for x in nr[:4]]).lower()
                        kws = self.TYPE_KEYWORDS.get(table_type, [])
                        hit = sum(1 for k in kws if k in joined)
                        score = float(len(nr)) + (1.5 * hit)
                        candidates.append((score, page_num, nh, nr))

        if not candidates:
            return [], []

        candidates.sort(key=lambda x: (x[0], len(x[3])), reverse=True)
        seed_score, seed_page, seed_h, seed_r = candidates[0]

        out_tables: List[Tuple[List[str], List[List[str]]]] = [(seed_h, seed_r)]
        out_pages: List[int] = [seed_page]

        # Include adjacent continuation pages for fallback as well.
        page_best: Dict[int, Tuple[float, List[str], List[List[str]]]] = {}
        for sc, p, h, r in candidates:
            prev = page_best.get(p)
            if prev is None or sc > prev[0]:
                page_best[p] = (sc, h, r)

        for p in [seed_page - 1, seed_page + 1]:
            if p in page_best:
                _, ch, cr = page_best[p]
                if self._is_continuation_candidate(table_type, seed_h, seed_r, ch, cr):
                    if p < seed_page:
                        out_tables.insert(0, (ch, cr))
                        out_pages.insert(0, p)
                    else:
                        out_tables.append((ch, cr))
                        out_pages.append(p)

        return out_tables, out_pages

    # --------------------------- Core extraction ---------------------------

    def _extract_pdfplumber_tables(
        self,
        pdf_path: str,
        table_type: str,
        question: str = "",
    ) -> Tuple[List[Tuple[List[str], List[List[str]]]], List[int]]:
        try:
            import pdfplumber
        except Exception:
            return [], []

        q_words = set(re.findall(r"[a-z]{3,}", (question or "").lower()))

        # Best candidate per page after normalization.
        page_best: Dict[int, Tuple[float, List[str], List[List[str]]]] = {}

        settings_candidates = [
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            },
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "intersection_tolerance": 3,
            },
            None,
        ]

        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_text = (page.extract_text() or "").lower()

                    for settings in settings_candidates:
                        try:
                            raw_tables = page.extract_tables(settings) if settings else (page.extract_tables() or [])
                        except Exception:
                            raw_tables = []

                        for raw in raw_tables or []:
                            headers, rows = self._normalize_raw_table(raw)
                            if not headers or not rows:
                                continue

                            # Normalize by requested type before scoring. This is key for reliable stitching.
                            headers, rows = self._postprocess_table_for_type(headers, rows, table_type)
                            if not headers or not rows:
                                continue

                            if not self._is_viable_typed_table(headers, rows, table_type):
                                continue

                            score = self._candidate_score(headers, rows, page_text, table_type, q_words)
                            prev = page_best.get(page_num)
                            if prev is None or score > prev[0]:
                                page_best[page_num] = (score, headers, rows)
        except Exception:
            return [], []

        if not page_best:
            return [], []

        # Seed page: highest score with most rows as tie-breaker.
        ranked = [
            (score, page, headers, rows)
            for page, (score, headers, rows) in page_best.items()
        ]
        ranked.sort(key=lambda x: (x[0], len(x[3])), reverse=True)
        seed_score, seed_page, seed_headers, seed_rows = ranked[0]

        if table_type != "generic" and seed_score <= 0.0:
            return [], []

        selected: List[Tuple[List[str], List[List[str]]]] = [(seed_headers, seed_rows)]
        selected_pages: List[int] = [seed_page]

        # Backward stitch: include immediate prior page(s) if they are true continuations.
        for prev_page in range(seed_page - 1, max(seed_page - 3, 0), -1):
            if prev_page not in page_best:
                break

            _, cand_h, cand_r = page_best[prev_page]
            if not self._is_continuation_candidate(table_type, seed_headers, seed_rows, cand_h, cand_r):
                break

            selected.insert(0, (cand_h, cand_r))
            selected_pages.insert(0, prev_page)

        # PDF stitch: extend only to immediate forward pages (N+1, N+2).
        for next_page in range(seed_page + 1, seed_page + 3):
            if next_page not in page_best:
                break

            _, cand_h, cand_r = page_best[next_page]
            if not self._is_continuation_candidate(table_type, seed_headers, seed_rows, cand_h, cand_r):
                break

            selected.append((cand_h, cand_r))
            selected_pages.append(next_page)

        return selected, selected_pages

    def _candidate_score(
        self,
        headers: List[str],
        rows: List[List[str]],
        page_text: str,
        table_type: str,
        q_words: set,
    ) -> float:
        joined = " ".join(headers + [" ".join(r) for r in rows[:6]]).lower()
        keywords = self.TYPE_KEYWORDS.get(table_type, [])
        rivals = self.RIVAL_KEYWORDS.get(table_type, [])

        own_hits = sum(1 for kw in keywords if kw in joined)
        rival_hits = sum(1 for kw in rivals if kw in joined)
        q_overlap = sum(1 for w in q_words if w in page_text or w in joined)
        avg_fill = sum(self._row_fill_ratio(r) for r in rows) / max(1, len(rows))
        dot_leader_rows = sum(1 for r in rows if self._is_dot_leader_row(r))
        fragmented_rows = sum(1 for r in rows if self._is_fragmented_row(r))
        exact_header_hits = sum(1 for kw in keywords if kw in " ".join(headers).lower())

        score = (own_hits * 3.0) - (rival_hits * 2.5) + (q_overlap * 0.2) + (avg_fill * 2.0) + (len(rows) * 0.05)
        score -= dot_leader_rows * 2.0
        score -= fragmented_rows * 3.0
        score += exact_header_hits * 1.5
        return score

    def _is_continuation_candidate(
        self,
        table_type: str,
        seed_headers: List[str],
        seed_rows: List[List[str]],
        cand_headers: List[str],
        cand_rows: List[List[str]],
    ) -> bool:
        if not cand_rows:
            return False

        sim = self._header_similarity(seed_headers, cand_headers)
        avg_fill = sum(self._row_fill_ratio(r) for r in cand_rows) / max(1, len(cand_rows))

        if table_type == "raci":
            # Prevent SIPOC bleed into RACI continuation.
            joined = " ".join(cand_headers + [" ".join(r) for r in cand_rows[:4]]).lower()
            sipoc_hits = sum(1 for kw in ["supplier", "input", "output", "customer", "sipoc"] if kw in joined)
            if sipoc_hits >= 2:
                return False
            return len(cand_rows) >= 1 and avg_fill >= 0.20

        if table_type == "sipoc":
            joined = " ".join(cand_headers + [" ".join(r) for r in cand_rows[:4]]).lower()
            raci_hits = sum(1 for kw in ["responsible", "accountable", "consulted", "informed", "raci"] if kw in joined)
            if raci_hits >= 2:
                return False
            return len(cand_rows) >= 1 and avg_fill >= 0.20

        # Generic continuation: require some structural similarity.
        return (sim >= 0.2 and avg_fill >= 0.25 and len(cand_rows) >= 1)

    # -------------------------- Merge and cleanup --------------------------

    def _merge_multipage_tables(
        self,
        tables: List[Tuple[List[str], List[List[str]]]],
        pages: List[int],
    ) -> Tuple[List[str], List[List[str]], List[int]]:
        if not tables:
            return [], [], []

        base_headers = tables[0][0]
        width = len(base_headers)

        merged_rows: List[List[str]] = []
        merged_pages: List[int] = []
        seen = set()

        for (headers, rows), page in zip(tables, pages):
            # Re-map rows to base width where possible.
            if len(headers) != width:
                rows_aligned = [self._align_row(r, width) for r in rows]
            else:
                rows_aligned = [list(r) for r in rows]

            for r in rows_aligned:
                rr = [self._clean_cell(c) for c in self._align_row(r, width)]
                if not any(rr):
                    continue
                key = tuple(x.lower().strip() for x in rr)
                if key in seen:
                    continue
                seen.add(key)
                merged_rows.append(rr)

            if page not in merged_pages:
                merged_pages.append(page)

        return base_headers, merged_rows, merged_pages

    def _postprocess_table_for_type(
        self,
        headers: List[str],
        rows: List[List[str]],
        table_type: str,
    ) -> Tuple[List[str], List[List[str]]]:
        if not headers or not rows:
            return headers, rows

        if table_type == "raci":
            headers, rows = self._normalize_raci_table(headers, rows)
        elif table_type == "sipoc":
            headers, rows = self._normalize_sipoc_table(headers, rows)

        width = len(headers)
        hnorm = [self._normalize_text(h) for h in headers]
        out_rows: List[List[str]] = []

        for r in rows:
            rr = self._align_row([self._clean_cell(c) for c in r], width)
            if not any(rr):
                continue
            if self._is_metadata_row(rr):
                continue
            if [self._normalize_text(c) for c in rr] == hnorm:
                continue

            # Drop split-header remainder rows like ["", "", "e", ...].
            long_cells = sum(1 for c in rr if len(c.strip()) > 1)
            if long_cells == 0:
                continue
            if self._is_fragmented_row(rr):
                continue

            out_rows.append(rr)

        return headers, out_rows

    def _normalize_raw_table(self, raw_table: Any) -> Tuple[List[str], List[List[str]]]:
        if not raw_table or len(raw_table) < 2:
            return [], []

        headers = [self._clean_cell(c) for c in raw_table[0]]
        if not any(headers):
            return [], []

        width = len(headers)
        rows: List[List[str]] = []
        for raw_row in raw_table[1:]:
            if raw_row is None:
                continue
            row = [self._clean_cell(c) for c in raw_row]
            row = self._align_row(row, width)
            if not any(row):
                continue
            rows.append(row)

        return headers, rows

    # --------------------------- Type normalizers --------------------------

    def _normalize_raci_table(self, headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        canonical = ["Activity", "Responsible", "Accountable", "Consulted", "Informed"]
        if not headers or not rows:
            return canonical, []

        width = len(headers)
        aligned_rows = [self._align_row([self._clean_cell(c) for c in r], width) for r in rows]

        # Common PDF artifact: header fragments split across header row + first data row.
        # Example: "Responsibl" in header and "e" in first body row.
        if aligned_rows:
            first = aligned_rows[0]
            single_letter_count = sum(1 for c in first if len(c.strip()) == 1 and c.strip().isalpha())
            if single_letter_count >= 2:
                rebuilt: List[str] = []
                for h, c in zip(headers, first):
                    hh = self._clean_cell(h)
                    cc = self._clean_cell(c)
                    if hh and len(cc) == 1 and cc.isalpha() and hh[-1:].isalpha():
                        rebuilt.append(hh + cc)
                    else:
                        rebuilt.append(hh)
                headers = rebuilt
                aligned_rows = aligned_rows[1:]

        role_idx = self._find_table_columns(headers, ["responsible", "accountable", "consulted", "informed"])

        # Fallback for initials-only headers: R | A | C | I
        if sum(1 for i in role_idx if i is not None) < 3:
            initials_idx = self._find_table_columns(headers, ["r", "a", "c", "i"])
            if sum(1 for i in initials_idx if i is not None) >= 3:
                role_idx = initials_idx

        # Fallback: continuation page often has no role headers but exactly 5 columns in RACI order.
        if sum(1 for i in role_idx if i is not None) < 3 and width == 5:
            out: List[List[str]] = []

            # pdfplumber may place the first continuation data row into headers.
            # Recover that row when header cells do not look like role labels.
            role_words = {"responsible", "accountable", "consulted", "informed", "activity", "raci"}
            header_row_candidate = [self._clean_cell(c).strip() for c in headers[:5]]
            header_joined = " ".join(header_row_candidate).lower()
            header_looks_like_role_labels = any(w in header_joined for w in role_words)
            if not header_looks_like_role_labels:
                if header_row_candidate[0] and sum(1 for c in header_row_candidate[1:] if c) >= 1:
                    if not self._is_metadata_row(header_row_candidate) and not self._is_toc_like_raci_row(header_row_candidate):
                        out.append(header_row_candidate)

            for r in aligned_rows:
                if self._is_metadata_row(r):
                    continue
                rr = [r[i].strip() for i in range(5)]
                if not rr[0] or sum(1 for c in rr[1:] if c) == 0:
                    continue
                if self._is_toc_like_raci_row(rr):
                    continue
                out.append(rr)
            return canonical, out

        # Probe body rows for role header row if parsing shifted them into row data.
        if sum(1 for i in role_idx if i is not None) < 3:
            for i, probe in enumerate(aligned_rows[: min(6, len(aligned_rows))]):
                guessed = self._find_table_columns(probe, ["responsible", "accountable", "consulted", "informed"])
                if sum(1 for x in guessed if x is not None) >= 3:
                    role_idx = guessed
                    aligned_rows = aligned_rows[i + 1 :]
                    headers = probe
                    width = len(headers)
                    break

        if sum(1 for i in role_idx if i is not None) < 3:
            return headers, rows

        role_src_idx = self._realign_shifted_role_columns(role_idx, headers, aligned_rows)
        activity_idx = self._pick_activity_column(headers, role_src_idx, aligned_rows)

        out_rows: List[List[str]] = []
        sipoc_terms = {"supplier", "input", "process", "output", "customer", "sipoc"}

        for r in aligned_rows:
            mapped = [r[activity_idx] if activity_idx < len(r) else ""]
            mapped.extend(r[i] if i is not None and i < len(r) else "" for i in role_src_idx)
            mapped = [self._clean_cell(c) for c in mapped]

            if self._is_metadata_row(mapped):
                continue
            if not mapped[0].strip():
                continue
            if sum(1 for c in mapped[1:] if c.strip()) == 0:
                continue
            if self._is_toc_like_raci_row(mapped):
                continue

            joined = " ".join(mapped).lower()
            if sum(1 for t in sipoc_terms if t in joined) >= 2:
                # stop when schema switches to SIPOC on same page.
                break

            out_rows.append(mapped)

        return canonical, out_rows

    def _normalize_sipoc_table(self, headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        if not headers or not rows:
            return ["Supplier", "Input", "Process", "Output", "Customer"], []

        width = len(headers)
        aligned_rows = [self._align_row([self._clean_cell(c) for c in r], width) for r in rows]
        norm_headers = [self._normalize_text(h) for h in headers]

        supplier_idx = self._find_header_idx(norm_headers, ["supplier"])
        input_idx = self._find_header_idx(norm_headers, ["input"])
        process_idx = self._find_header_idx(norm_headers, ["process"])
        output_idx = self._find_header_idx(norm_headers, ["output"])
        customer_idx = self._find_header_idx(norm_headers, ["customer"])

        if sum(1 for i in [supplier_idx, input_idx, process_idx, output_idx, customer_idx] if i is not None) < 3:
            return headers, rows

        idx = [supplier_idx, input_idx, process_idx, output_idx, customer_idx]
        canonical = ["Supplier", "Input", "Process", "Output", "Customer"]

        out_rows: List[List[str]] = []
        for r in aligned_rows:
            mapped = [r[i] if i is not None and i < len(r) else "" for i in idx]

            # Recover missing Process when header is a merged "Process and Control" block.
            if not str(mapped[2]).strip() and input_idx is not None and output_idx is not None:
                lo = min(input_idx, output_idx) + 1
                hi = max(input_idx, output_idx)
                middle_vals = [
                    r[c].strip()
                    for c in range(lo, hi)
                    if c < len(r) and r[c] and r[c].strip()
                ]
                if middle_vals:
                    mapped[2] = max(middle_vals, key=len)

            if self._is_metadata_row(mapped):
                continue
            if sum(1 for c in mapped if c.strip()) < 2:
                continue
            out_rows.append(mapped)

        return canonical, out_rows

    def _normalize_change_history_table(self, headers: List[str], rows: List[List[str]]) -> Tuple[List[str], List[List[str]]]:
        if not headers or not rows:
            return headers, rows

        width = len(headers)
        out: List[List[str]] = []

        for r in rows:
            rr = self._align_row([self._clean_cell(c) for c in r], width)
            if self._is_metadata_row(rr):
                continue

            first = rr[0].strip().lower() if rr else ""
            if not first:
                # continuation row: append to previous change row
                if out:
                    prev = out[-1]
                    for i in range(width):
                        if rr[i]:
                            prev[i] = (prev[i] + " " + rr[i]).strip() if prev[i] else rr[i]
                    out[-1] = prev
                continue

            if re.match(r"^\d{1,3}([\.)])?$", first):
                out.append(rr)

        return headers, out

    # ------------------------------ Utilities ------------------------------

    def _find_table_columns(self, headers: List[str], keys: List[str]) -> List[Optional[int]]:
        norm_headers = [self._normalize_text(h) for h in headers]
        compact_headers = [self._compact_token(h) for h in headers]
        used = set()
        out: List[Optional[int]] = []

        for key in keys:
            pick = None
            key_compact = self._compact_token(key)
            for i, h in enumerate(norm_headers):
                if i in used:
                    continue
                if len(key) == 1:
                    # For initials (R/A/C/I), require exact token-level match.
                    tokens = set(h.split())
                    if key in tokens or compact_headers[i] == key:
                        pick = i
                        break
                elif key in h or (key_compact and key_compact in compact_headers[i]):
                    pick = i
                    break
            if pick is not None:
                used.add(pick)
            out.append(pick)
        return out

    def _realign_shifted_role_columns(
        self,
        role_idx: List[Optional[int]],
        headers: List[str],
        rows: List[List[str]],
    ) -> List[Optional[int]]:
        if not rows:
            return role_idx

        width = len(headers)

        def fill_ratio(col_idx: int) -> float:
            non_empty = sum(1 for r in rows if col_idx < len(r) and str(r[col_idx]).strip())
            return non_empty / max(1, len(rows))

        out: List[Optional[int]] = []
        for idx in role_idx:
            if idx is None or idx >= width:
                out.append(idx)
                continue

            cur_fill = fill_ratio(idx)
            if idx > 0:
                left_fill = fill_ratio(idx - 1)
                if cur_fill <= 0.20 and left_fill >= 0.35:
                    out.append(idx - 1)
                    continue
            out.append(idx)

        return out

    def _pick_activity_column(
        self,
        headers: List[str],
        role_idx: List[Optional[int]],
        rows: Optional[List[List[str]]] = None,
    ) -> int:
        used = {i for i in role_idx if i is not None}
        activity_words = ["activity", "process", "task", "step", "function", "description"]
        norm_headers = [self._normalize_text(h) for h in headers]
        width = len(headers)
        aligned_rows = [self._align_row(r, width) for r in (rows or [])] if rows else []

        def fill_ratio(col_idx: int) -> float:
            if not aligned_rows:
                return 0.0
            non_empty = sum(1 for r in aligned_rows if col_idx < len(r) and str(r[col_idx]).strip())
            return non_empty / max(1, len(aligned_rows))

        for i, h in enumerate(norm_headers):
            if i in used:
                continue
            if any(w in h for w in activity_words):
                if i > 0 and (i - 1) not in used and fill_ratio(i) <= 0.2 and fill_ratio(i - 1) >= 0.35:
                    return i - 1
                return i

        best_i = None
        best_fill = -1.0
        for i in range(len(headers)):
            if i in used:
                continue
            f = fill_ratio(i)
            if f > best_fill:
                best_fill = f
                best_i = i
        if best_i is not None:
            return best_i

        for i in range(len(headers)):
            if i not in used:
                return i
        return 0

    def _find_header_idx(self, norm_headers: List[str], keywords: List[str]) -> Optional[int]:
        for i, h in enumerate(norm_headers):
            if any(k in h for k in keywords):
                return i
        return None

    def _is_viable_typed_table(self, headers: List[str], rows: List[List[str]], table_type: str) -> bool:
        if not headers or not rows:
            return False

        joined = " ".join(headers + [" ".join(r) for r in rows[:4]]).lower()

        if table_type == "raci":
            if headers == ["Activity", "Responsible", "Accountable", "Consulted", "Informed"]:
                good = sum(
                    1
                    for r in rows
                    if (
                        r
                        and str(r[0]).strip()
                        and not self._is_toc_like_raci_row(r)
                        and sum(1 for c in r[1:] if str(c).strip() and not self._is_dot_leader_text(str(c))) >= 1
                    )
                )
                return good >= 2
            return any(k in joined for k in ["responsible", "accountable", "consulted", "informed", "raci"])

        if table_type == "sipoc":
            if len(headers) < 5 or len(rows) < 1:
                return False
            header_text = " ".join(headers).lower()
            header_hits = sum(1 for kw in ["supplier", "input", "process", "output", "customer"] if kw in header_text)
            if header_hits < 3:
                return False
            fragmented = sum(1 for r in rows if self._is_fragmented_row(r))
            if fragmented > max(1, int(len(rows) * 0.5)):
                return False
            good_rows = 0
            for r in rows:
                non_empty = sum(1 for c in r if str(c).strip())
                long_cells = sum(1 for c in r if len(str(c).strip()) >= 3)
                if non_empty >= 3 and long_cells >= 2 and not self._is_fragmented_row(r):
                    good_rows += 1
            return good_rows >= 1

        if table_type == "change_history":
            return any(re.match(r"^\d{1,3}([\.)])?$", str(r[0]).strip()) for r in rows if r)

        return len(headers) >= 2 and len(rows) >= 1

    def _extract_camelot_tables(
        self,
        pdf_path: str,
        table_type: str,
        question: str = "",
    ) -> Tuple[List[Tuple[List[str], List[List[str]]]], List[int]]:
        # Optional fallback intentionally disabled by default for stability.
        return [], []

    def _to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        safe_headers = [self._normalize_output_text(h).replace("|", "\\|") for h in headers]
        lines = [
            "| " + " | ".join(safe_headers) + " |",
            "| " + " | ".join(["---"] * len(safe_headers)) + " |",
        ]
        for r in rows:
            rr = [self._normalize_output_text(c).replace("|", "\\|") for c in self._align_row(r, len(headers))]
            lines.append("| " + " | ".join(rr) + " |")
        return "\n".join(lines)

    def _align_row(self, row: List[str], width: int) -> List[str]:
        if len(row) < width:
            return row + [""] * (width - len(row))
        if len(row) > width:
            return row[:width]
        return row

    def _clean_cell(self, value: Any) -> str:
        text = str(value or "")
        text = text.replace("\n", " ").replace("\t", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _normalize_output_text(self, value: Any) -> str:
        text = self._clean_cell(value)
        if not text:
            return ""
        text = self._fix_split_words(text)
        text = self._fix_spaced_letters(text)
        return text

    def _fix_split_words(self, text: str) -> str:
        text = re.sub(r"\b([A-Za-z]{1,4})\s+([A-Za-z]{1,2})\b", r"\1\2", text)
        text = re.sub(r"\b([A-Za-z]{4,})\s+([a-z])\b", r"\1\2", text)
        return text

    def _fix_spaced_letters(self, text: str) -> str:
        def repl(match: re.Match) -> str:
            return match.group(0).replace(" ", "")

        return re.sub(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", repl, text)

    def _is_metadata_row(self, row: List[str]) -> bool:
        txt = " ".join(str(c).strip() for c in row if str(c).strip()).lower()
        if not txt:
            return True

        if any(re.search(p, txt) for p in [
            r"document no\s*:", r"document title\s*:", r"effective date\s*:",
            r"next review\s*date\s*:", r"document status\s*:", r"document classification\s*:",
            r"document template\s*:", r"^page\s*\d+", r"\bnda\b",
        ]):
            return True

        word_count = len(txt.split())
        if word_count > 10:
            return False
        return any(re.search(p, txt) for p in self.meta_patterns)

    def _row_fill_ratio(self, row: List[str]) -> float:
        if not row:
            return 0.0
        non_empty = sum(1 for c in row if str(c).strip() and str(c).strip().lower() not in {"n/a", "na"})
        return non_empty / max(1, len(row))

    def _header_similarity(self, h1: List[str], h2: List[str]) -> float:
        a = set(self._normalize_text(x) for x in h1 if str(x).strip())
        b = set(self._normalize_text(x) for x in h2 if str(x).strip())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _is_dot_leader_text(self, text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        if re.search(r"\.{3,}", t):
            return True
        if re.fullmatch(r"[.\s\d]+", t):
            return True
        return False

    def _is_dot_leader_row(self, row: List[str]) -> bool:
        if not row:
            return False
        hits = sum(1 for c in row if self._is_dot_leader_text(str(c)))
        return hits >= max(2, int(len(row) * 0.4))

    def _is_toc_like_raci_row(self, row: List[str]) -> bool:
        if not row or len(row) < 5:
            return False

        activity = str(row[0]).strip().lower()
        role_cells = [str(c).strip() for c in row[1:]]
        dot_cells = sum(1 for c in role_cells if self._is_dot_leader_text(c))

        # Typical table-of-contents line: "1 Purpose ...." with dot leaders and page number.
        if re.match(r"^\d+\s+[a-z]", activity) and dot_cells >= 2:
            return True
        if dot_cells >= 3:
            return True
        return False

    def _is_fragmented_row(self, row: List[str]) -> bool:
        vals = [str(c).strip() for c in row if str(c).strip()]
        if not vals:
            return False
        tiny = sum(1 for v in vals if len(v) <= 2)
        alpha_tiny = sum(1 for v in vals if len(v) <= 2 and re.search(r"[a-zA-Z]", v))
        return tiny >= max(3, int(len(vals) * 0.6)) or alpha_tiny >= 3

    def _normalize_text(self, text: str) -> str:
        lowered = (text or "").lower()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
        return " ".join(cleaned.split())

    def _compact_token(self, text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (text or "").lower())

    def _table_type(self, question: str) -> str:
        q = (question or "").lower()
        if "raci" in q:
            return "raci"
        if "sipoc" in q:
            return "sipoc"
        return "generic"

    def _match_pdf_file(self, question: str) -> Optional[str]:
        q = self._normalize_text(question)
        if not q:
            return None

        root = Path(self.pdf_dir)
        if not root.exists():
            return None

        best = None
        best_len = 0
        q_words = set(q.split())

        for pdf in root.glob("*.pdf"):
            stem = self._normalize_text(pdf.stem)
            if stem and stem in q and len(stem) > best_len:
                best = pdf.name
                best_len = len(stem)
                continue

            stem_words = set(stem.split()) - {"sop", "ut", "of", "and", "the", "in", "for", "a", "an", "to"}
            overlap = stem_words & q_words
            if stem_words and len(overlap) >= max(2, int(len(stem_words) * 0.5)) and len(overlap) > best_len:
                best = pdf.name
                best_len = len(overlap)

        return best

    def _find_pdf_by_table_type(self, table_type: str) -> Optional[str]:
        root = Path(self.pdf_dir)
        if not root.exists():
            return None

        # Prefer data-driven choice from scanned table catalog.
        if table_type in {"raci", "sipoc"}:
            catalog = self._load_or_build_table_catalog(force=False)
            docs = catalog.get("documents", {}) if isinstance(catalog, dict) else {}
            best_doc = None
            best_score = -1
            for doc_name, meta in docs.items():
                tmeta = (meta or {}).get(table_type, {})
                score = int(tmeta.get("score", 0))
                pages = tmeta.get("pages", [])
                if score > best_score and pages:
                    best_score = score
                    best_doc = doc_name
            if best_doc:
                return best_doc

        keywords = self.TYPE_KEYWORDS.get(table_type, [])
        if not keywords:
            first = next(root.glob("*.pdf"), None)
            return first.name if first else None

        best_name = None
        best_score = -1

        for pdf in root.glob("*.pdf"):
            stem = self._normalize_text(pdf.stem)
            score = sum(1 for kw in keywords if kw in stem)
            if score > best_score:
                best_score = score
                best_name = pdf.name

        if best_name:
            return best_name

        first = next(root.glob("*.pdf"), None)
        return first.name if first else None

    def _load_or_build_table_catalog(self, force: bool = False) -> Dict[str, Any]:
        if self._table_catalog_cache is not None and not force:
            return self._table_catalog_cache

        root = Path(self.pdf_dir)
        if not root.exists():
            self._table_catalog_cache = {"documents": {}, "pdf_count": 0}
            return self._table_catalog_cache

        pdf_files = sorted(root.glob("*.pdf"))
        pdf_count = len(pdf_files)

        if (not force) and os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if int(cached.get("pdf_count", -1)) == pdf_count:
                    self._table_catalog_cache = cached
                    return cached
            except Exception:
                pass

        catalog = {
            "pdf_count": pdf_count,
            "documents": {},
        }

        try:
            import pdfplumber
        except Exception:
            self._table_catalog_cache = catalog
            return catalog

        settings_candidates = [
            {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "intersection_tolerance": 5,
            },
            {
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "intersection_tolerance": 3,
            },
            None,
        ]

        for pdf in pdf_files:
            per_type_pages: Dict[str, set] = {"raci": set(), "sipoc": set()}
            per_type_score: Dict[str, int] = {"raci": 0, "sipoc": 0}
            pdf_path = str(pdf)

            try:
                with pdfplumber.open(pdf_path) as doc:
                    for page_num, page in enumerate(doc.pages, start=1):
                        seen_on_page = {"raci": False, "sipoc": False}
                        for settings in settings_candidates:
                            try:
                                raw_tables = page.extract_tables(settings) if settings else (page.extract_tables() or [])
                            except Exception:
                                raw_tables = []

                            for raw in raw_tables or []:
                                h, r = self._normalize_raw_table(raw)
                                if not h or not r:
                                    continue

                                for t in ("raci", "sipoc"):
                                    nh, nr = self._postprocess_table_for_type(h, r, t)
                                    if not nh or not nr:
                                        continue
                                    if self._is_viable_typed_table(nh, nr, t):
                                        per_type_pages[t].add(page_num)
                                        if not seen_on_page[t]:
                                            per_type_score[t] += max(1, len(nr))
                                            seen_on_page[t] = True
            except Exception:
                pass

            catalog["documents"][pdf.name] = {
                "raci": {
                    "pages": sorted(per_type_pages["raci"]),
                    "score": per_type_score["raci"],
                },
                "sipoc": {
                    "pages": sorted(per_type_pages["sipoc"]),
                    "score": per_type_score["sipoc"],
                },
            }

        try:
            with open(self.catalog_path, "w", encoding="utf-8") as f:
                json.dump(catalog, f, ensure_ascii=True, indent=2)
        except Exception:
            pass

        self._table_catalog_cache = catalog
        return catalog
