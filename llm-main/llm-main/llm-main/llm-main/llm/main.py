import argparse
from src.rag_pipeline import SOPRagPipeline

def main():
    parser = argparse.ArgumentParser(description="Quality SOP RAG System")
    parser.add_argument("--index", action="store_true", help="Index the PDF documents")
    parser.add_argument("--query", type=str, help="Ask a question about the SOPs")
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="pdfs",
        help="Path to the folder containing PDF documents",
    )
    parser.add_argument("--provider", type=str, default="", help="LLM provider: groq or local")
    parser.add_argument("--model", type=str, default="", help="Groq model name")
    parser.add_argument("--base-url", type=str, default="", help="Groq API base URL")
    args = parser.parse_args()

    pipeline = SOPRagPipeline(
        pdf_dir=args.pdf_dir,
        llm_provider=args.provider,
        groq_model=args.model,
        groq_base_url=args.base_url,
    )

    if args.index:
        pipeline.load_and_process_documents()
    
    if args.query:
        print(f"\nQuestion: {args.query}")
        result = pipeline.answer_question(args.query)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {', '.join(result['sources'])}")

if __name__ == "__main__":
    main()
