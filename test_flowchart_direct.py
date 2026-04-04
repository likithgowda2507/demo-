#!/usr/bin/env python
"""Test flowchart extraction directly on Inventory Management SOP."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import SOPRagPipeline

# Initialize RAG
pdf_dir = r"d:\Documents\llm-main\llm-main\llm-main\llm-main\llm-main\llm\pdfs"
rag = SOPRagPipeline(pdf_dir=pdf_dir)

# Find the Inventory Management PDF
import glob
pdfs = glob.glob(os.path.join(pdf_dir, "*Inventory*"))
 
if pdfs:
    pdf_path = pdfs[0]  # Use first match
    print(f"Testing DIRECT flowchart extraction")
    print(f"PDF: {os.path.basename(pdf_path)}\n")
    print("="*60)
    
    result = rag._extract_flowchart_images_fast(
        pdf_filename=pdf_path,
        question="GRN process flowchart",
        max_images=1
    )
    
    print("\n" + "="*60)
    print("RESULT:")
    print("="*60)
    
    if result:
        print(f"✓ Found {len(result)} flowchart image(s)")
        for i, img in enumerate(result):
            print(f"  - Image {i+1}: {img.filename}")
    else:
        print(f"✗ No flowchart images found")
else:
    print("ERROR: No Inventory Management PDF found")
    print("Available PDFs:")
    for f in sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))[:15]:
        print(f"  - {os.path.basename(f)}")
