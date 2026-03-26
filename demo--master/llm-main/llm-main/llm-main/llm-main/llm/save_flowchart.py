#!/usr/bin/env python
"""Save extracted flowchart image to disk."""

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
    pdf_path = pdfs[0]
    print(f"Extracting flowchart from: {os.path.basename(pdf_path)}\n")
    
    result = rag._extract_flowchart_images_fast(
        pdf_filename=pdf_path,
        question="GRN process flowchart",
        max_images=1
    )
    
    if result:
        # Save the image
        output_dir = os.path.join(os.path.dirname(__file__), 'flowchart_snapshots')
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, 'grn_flowchart_test.png')
        with open(output_file, 'wb') as f:
            f.write(result[0])  # result[0] is the bytes data
        
        print(f"[OK] Flowchart saved to: {output_file}")
        print(f"     File size: {len(result[0])} bytes")
    else:
        print(f"[FAILED] No flowchart images found")
else:
    print("ERROR: No Inventory Management PDF found")
