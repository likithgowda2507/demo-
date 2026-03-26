#!/usr/bin/env python3
"""
Diagnostic script to test flowchart extraction setup.
Run this to identify what's preventing flowchart generation.
"""
import sys
from pathlib import Path

print("\n" + "="*60)
print("Flowchart Setup Diagnostics")
print("="*60 + "\n")

# Test 1: pdfplumber
print("[1/5] Testing pdfplumber...")
try:
    import pdfplumber
    print("✓ pdfplumber installed")
except ImportError as e:
    print(f"✗ pdfplumber NOT installed: {e}")
    sys.exit(1)

# Test 2: pdf2image
print("\n[2/5] Testing pdf2image...")
try:
    from pdf2image import convert_from_path
    print("✓ pdf2image installed")
except ImportError as e:
    print(f"✗ pdf2image NOT installed: {e}")
    sys.exit(1)

# Test 3: Poppler
print("\n[3/5] Testing Poppler (system dependency)...")
try:
    from pdf2image import convert_from_path
    # Quick test
    test_pdf = Path("llm-main/llm-main/llm-main/llm-main/llm/pdfs").glob("*.pdf")
    test_pdf = list(test_pdf)
    
    if test_pdf:
        print(f"✓ Poppler appears available (found {len(test_pdf)} PDFs)")
    else:
        print("⚠ No PDFs found in pdfs folder, but Poppler likely installed")
except Exception as e:
    print(f"✗ Poppler NOT installed or issue: {e}")
    print("  Install with: choco install poppler (Windows)")
    sys.exit(1)

# Test 4: Transformers + Donut
print("\n[4/5] Testing transformers (for Donut model)...")
try:
    from transformers import DonutProcessor, VisionEncoderDecoderModel
    print("✓ transformers installed")
except ImportError as e:
    print(f"✗ transformers NOT installed: {e}")
    sys.exit(1)

# Test 5: PyTorch
print("\n[5/5] Testing PyTorch...")
try:
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ PyTorch installed (using {device})")
except ImportError as e:
    print(f"✗ PyTorch NOT installed: {e}")
    sys.exit(1)

# Summary
print("\n" + "="*60)
print("All dependencies OK!")
print("="*60)
print("\nNext steps:")
print("1. If still hanging, check your PDF files exist in llm-main/llm-main/llm-main/llm-main/llm/pdfs/")
print("2. Ensure PDFs contain visible flowcharts or 'flowchart'/'GRN'/'process' keywords")
print("3. On Windows, Poppler must be installed (choco install poppler)")
print("4. First flowchart generation is slow (Donut model loads and downloads)")
print("5. Check console output for [FLOWCHART] diagnostic messages")
print("="*60 + "\n")
