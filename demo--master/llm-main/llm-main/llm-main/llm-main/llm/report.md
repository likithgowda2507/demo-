# Project Report: Quality SOP RAG Pipeline

## Approach
The goal was to build a robust, locally runnable RAG pipeline for querying sensitive Quality SOP documents. 

To achieve this:
- **Local LLM**: I chose the `LaMini-T5-738M` model. It is small enough to run on most CPUs while being fine-tuned for instruction following, making it ideal for summarizing SOP context without requiring external API access (privacy-friendly).
- **Modern LangChain (LCEL)**: The implementation uses LangChain Expression Language (LCEL). This ensures better compatibility with the latest LangChain versions and provides a cleaner, more modular way to define the retrieval-generation chain.
- **Efficient Retrieval**: FAISS was used for the vector store due to its speed and low overhead for local file-based storage.

## Limitations
1.  **Model Size**: While `LaMini-T5` is efficient, complex reasoning might be limited compared to larger models like Llama-3 or GPT-4.
2.  **Context Window**: T5 models have a relatively small context window (512 tokens). This is why chunking and retrieving only the top 3 chunks is critical.
3.  **Table Extraction**: Standard PDF loaders can struggle with complex tables. If the SOPs rely heavily on tabular data, a more specialized table extraction tool (like `unstructured`) might be needed.
4.  **Hardware Dependencies**: While it runs on CPU, indexing 74 documents can take a few minutes. GPU acceleration would significantly speed up embedding generation.
