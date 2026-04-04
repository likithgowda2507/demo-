"""
Table Extraction Debugger - run with:
  python test_table_debug.py "your question here"

Shows exactly which pages/rows are being extracted and why.
"""
import sys
from pathlib import Path

PDF_DIR = str(Path(__file__).parent / "pdfs")
sys.path.insert(0, str(Path(__file__).parent / "src"))

from table_extractor import TableExtractor

question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "give me the change history table"
print(f"\n=== Table Extraction Debug ===")
print(f"Question: {question}\n")

extractor = TableExtractor(pdf_dir=PDF_DIR)
table_type = extractor._table_type(question)
print(f"Detected table type: {table_type}")

matched = extractor._match_pdf_file(question)
print(f"Matched PDF: {matched or 'None'}")

if not matched:
    matched = extractor._find_pdf_by_table_type(table_type)
    print(f"Auto-found PDF by table type: {matched or 'None'}")

if not matched:
    print("ERROR: No PDF found!")
    sys.exit(1)

pdf_path = str(Path(PDF_DIR) / matched)

print(f"\n--- Running pdfplumber extraction ---")
tables, pages = extractor._extract_pdfplumber_tables(pdf_path, table_type, question)
print(f"Found {len(tables)} table(s) on pages: {pages}")

for i, ((headers, rows), page) in enumerate(zip(tables, pages)):
    print(f"\n  Table {i+1} (page {page}):")
    print(f"    Headers: {headers}")
    print(f"    Row count: {len(rows)}")
    for j, r in enumerate(rows[:10]):
        print(f"    Row {j+1}: {r}")
    if len(rows) > 10:
        print(f"    ... and {len(rows)-10} more rows")

if not tables:
    print("\n  pdfplumber found nothing. Trying camelot...")
    tables, pages = extractor._extract_camelot_tables(pdf_path, table_type, question)
    print(f"  Camelot found {len(tables)} table(s) on pages: {pages}")

if tables:
    print(f"\n--- Merging multipage tables ---")
    merged_h, merged_r, merged_p = extractor._merge_multipage_tables(tables, pages)
    print(f"Merged pages: {merged_p}")
    print(f"Merged headers: {merged_h}")
    print(f"Total merged rows: {len(merged_r)}")
    print("\nFirst 15 rows after merge:")
    for j, r in enumerate(merged_r[:15]):
        print(f"  Row {j+1}: {r}")

    print(f"\n--- Post-processing for type: {table_type} ---")
    final_h, final_r = extractor._postprocess_table_for_type(merged_h, merged_r, table_type)
    print(f"Final headers: {final_h}")
    print(f"Final row count: {len(final_r)}")
    print("\nFinal rows:")
    for j, r in enumerate(final_r):
        print(f"  Row {j+1}: {r}")
