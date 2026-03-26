#!/usr/bin/env python3
"""
Clear flowchart image cache to force fresh extraction on next request.
Run this after code changes to flowchart detection.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "llm-main/llm-main/llm-main/llm-main/llm/src"))

try:
    from src.rag_pipeline import _FLOWCHART_IMAGE_CACHE
    print(f"[CACHE] Current cache size: {len(_FLOWCHART_IMAGE_CACHE)} items")
    _FLOWCHART_IMAGE_CACHE.clear()
    print(f"✓ Cache cleared! Size now: {len(_FLOWCHART_IMAGE_CACHE)}")
except Exception as e:
    print(f"Note: Cache is in-memory. Restart Streamlit to clear it automatically.")
    print(f"Error: {e}")

print("\nTo force fresh flowchart extraction:")
print("1. Stop Streamlit (Ctrl+C)")
print("2. Restart: .\\venv\\Scripts\\python.exe -m streamlit run llm-main\\llm-main\\llm-main\\llm-main\\llm\\streamlit_app.py")
print("3. Try your flowchart request again")
