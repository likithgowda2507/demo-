# pyre-ignore-all-errors
"""
Flowchart Extraction Debugger
Run: python test_flowchart_debug.py "your question here"
"""
import sys
import importlib
from pathlib import Path

# Adjust this path if needed
PDF_DIR = str(Path(__file__).parent / "pdfs")
IMAGES_DIR = str(Path(__file__).parent / "flowchart_images")

question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "show me the flowchart"
print(f"\n=== Flowchart Debug ===")
print(f"Question: {question}\n")

sys.path.insert(0, str(Path(__file__).parent / "src"))
from flowchart_extractor import FlowchartExtractor

extractor = FlowchartExtractor(pdf_dir=PDF_DIR, images_dir=IMAGES_DIR)

# Show which PDF is matched
matched = extractor._match_pdf_file(question)
print(f"Matched PDF: {matched or 'None — will scan all PDFs'}")

# Show ranked PDF candidates
query_terms = extractor._query_terms(question)
print(f"Query terms: {query_terms}")
focus = extractor._focus_phrase(question)
print(f"Focus phrase: {focus!r}\n")

# Find all PDFs to scan
root = Path(PDF_DIR)
pdf_candidates = [matched] if matched else [p.name for p in sorted(root.glob("*.pdf"))]
print(f"PDFs to scan: {pdf_candidates}\n")

fitz = importlib.import_module("fitz")
keywords = ["flowchart", "flow chart", "process flow", "process flow chart",
            "overall process flow", "workflow", "diagram", "swimlane"]

for pdf_name in pdf_candidates[:3]:  # limit to first 3
    pdf_path = str(root / pdf_name)
    if not Path(pdf_path).exists():
        continue
    print(f"--- Scanning: {pdf_name} ---")
    doc = fitz.open(pdf_path)
    ranked = extractor._rank_flowchart_pages(doc, question, keywords, query_terms)
    print(f"  Top ranked pages (page_idx, score):")
    for page_idx, score in ranked[:5]:
        page = doc[page_idx]
        text = (page.get_text("text") or "")[:120].replace("\n", " ")
        ds = extractor._diagram_signal(fitz, page)
        is_likely = extractor._is_likely_flowchart_page(fitz, page, text)
        is_skip = extractor._is_skip_page(text, page_idx)
        print(f"    Page {page_idx+1}: score={score:.1f}, diagram_signal={ds}, likely_flowchart={is_likely}, skip={is_skip}")
        print(f"      Text preview: {text[:80]}...")
    doc.close()
    print()
