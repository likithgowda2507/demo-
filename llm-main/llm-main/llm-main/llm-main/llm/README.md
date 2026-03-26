# Quality SOP RAG Pipeline

A Retrieval-Augmented Generation (RAG) system built to answer questions based on Quality Standard Operating Procedure (SOP) documents.

## Architecture

The system follows a standard RAG architecture:
1.  **Document Ingestion**: Loads PDFs from the `pdfs/` directory using `PyPDFLoader`.
2.  **Text Chunking**: Splits text into chunks of 700 characters with a 100-character overlap using `RecursiveCharacterTextSplitter`.
3.  **Embedding Generation**: Converts text chunks into 384-dimensional embeddings using the `all-MiniLM-L6-v2` Sentence Transformer model.
4.  **Vector Store**: Stores embeddings in a local `FAISS` vector database for efficient similarity search.
5.  **Retrieval**: Uses similarity search to find the top 3 most relevant chunks for a user query.
6.  **Response Generation**: Uses a local LLM (`LaMini-T5-738M`) via LangChain's LCEL (LangChain Expression Language) to generate an answer based on the retrieved context.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Setup and Running

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Index the documents**:
    (This only needs to be run once or when new PDFs are added to the `pdfs/` folder)
    ```bash
    python3 main.py --index
    ```

3.  **Ask a question**:
    ```bash
    python3 main.py --query "What is the procedure for handling non-conforming products?"
    ```

## Example Query
- **Question**: "What is the procedure for handling non-conforming products?"
- **Answer**: "The procedure for handling non-conforming products is to identify and segregate them, notify the customer, and take corrective actions if necessary."
- **Sources**: SOP-Control of Non-Conforming Product (CNP)-UTPSCNPSOP.pdf, SOP-Incoming Goods Inspection UTPSIGISOP.pdf
