import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from src.rag_pipeline import SOPRagPipeline

HTML_PAGE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Quality SOP Chat</title>
    <style>
      :root {
        --bg: #0c1020;
        --panel: #141a2e;
        --panel-2: #0f1426;
        --accent: #18c29c;
        --accent-2: #3fd0ff;
        --text: #e6eefb;
        --muted: #a8b3d3;
        --danger: #ff6b6b;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: "Space Grotesk", "Trebuchet MS", "Segoe UI", sans-serif;
        color: var(--text);
        background: radial-gradient(1200px 800px at 10% 10%, #1a2445 0%, var(--bg) 55%, #0b0f1f 100%);
        min-height: 100vh;
        display: grid;
        place-items: center;
        padding: 24px;
      }
      .app {
        width: min(980px, 100%);
        background: linear-gradient(135deg, rgba(24,194,156,0.08), rgba(63,208,255,0.08)) , var(--panel);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.4);
        overflow: hidden;
      }
      header {
        padding: 20px 24px;
        background: var(--panel-2);
        display: flex;
        align-items: center;
        gap: 16px;
      }
      .badge {
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: #0b0f1f;
        font-weight: 700;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        letter-spacing: 0.5px;
      }
      h1 {
        font-size: 18px;
        margin: 0;
        letter-spacing: 0.3px;
      }
      .subtitle {
        color: var(--muted);
        font-size: 13px;
      }
      .chat {
        padding: 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
        max-height: 60vh;
        overflow-y: auto;
      }
      .msg {
        padding: 14px 16px;
        border-radius: 14px;
        max-width: 85%;
        line-height: 1.45;
        white-space: pre-wrap;
      }
      .msg.user {
        align-self: flex-end;
        background: linear-gradient(135deg, rgba(24,194,156,0.22), rgba(63,208,255,0.2));
        border: 1px solid rgba(63,208,255,0.4);
      }
      .msg.bot {
        align-self: flex-start;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
      }
      .sources {
        margin-top: 8px;
        color: var(--muted);
        font-size: 12px;
      }
      .composer {
        padding: 16px 24px 24px 24px;
        background: var(--panel-2);
        border-top: 1px solid rgba(255,255,255,0.08);
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 12px;
      }
      textarea {
        width: 100%;
        resize: none;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(10,14,26,0.9);
        color: var(--text);
        padding: 12px 14px;
        font-size: 14px;
        outline: none;
        min-height: 52px;
      }
      button {
        border: none;
        padding: 0 18px;
        border-radius: 12px;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: #0b0f1f;
        font-weight: 700;
        cursor: pointer;
        transition: transform 120ms ease, filter 120ms ease;
      }
      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }
      button:hover:not(:disabled) {
        transform: translateY(-1px);
        filter: brightness(1.05);
      }
      .hint {
        padding: 0 24px 20px 24px;
        color: var(--muted);
        font-size: 12px;
      }
      .error {
        color: var(--danger);
      }
      @media (max-width: 700px) {
        .chat { max-height: 55vh; }
        header { flex-direction: column; align-items: flex-start; }
      }
    </style>
  </head>
  <body>
    <div class="app">
      <header>
        <span class="badge">RAG</span>
        <div>
          <h1>Quality SOP Chat Assistant</h1>
          <div class="subtitle">Ask questions about your SOP PDFs</div>
        </div>
      </header>
      <div id="chat" class="chat"></div>
      <div class="composer">
        <textarea id="input" placeholder="Ask a question about non-conforming products, inspections, SOP steps..."></textarea>
        <button id="send">Send</button>
      </div>
      <div class="hint" id="status">Local model + FAISS index</div>
    </div>
    <script>
      const chat = document.getElementById("chat");
      const input = document.getElementById("input");
      const send = document.getElementById("send");
      const status = document.getElementById("status");

      function addMessage(text, role, sources) {
        const msg = document.createElement("div");
        msg.className = "msg " + role;
        msg.textContent = text;
        if (sources && sources.length) {
          const src = document.createElement("div");
          src.className = "sources";
          src.textContent = "Sources: " + sources.join(", ");
          msg.appendChild(src);
        }
        chat.appendChild(msg);
        chat.scrollTop = chat.scrollHeight;
      }

      async function sendMessage() {
        const text = input.value.trim();
        if (!text) return;
        addMessage(text, "user");
        input.value = "";
        send.disabled = true;
        status.textContent = "Thinking...";
        status.className = "hint";

        try {
          const res = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: text })
          });
          const data = await res.json();
          if (!res.ok) {
            throw new Error(data.error || "Request failed");
          }
          addMessage(data.answer || "No response.", "bot", data.sources || []);
          status.textContent = "Ready";
        } catch (err) {
          status.textContent = err.message;
          status.className = "hint error";
          addMessage("Sorry, I could not answer that. " + err.message, "bot");
        } finally {
          send.disabled = false;
        }
      }

      send.addEventListener("click", sendMessage);
      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          sendMessage();
        }
      });

      addMessage("Hi! Ask me anything about your SOP PDFs.", "bot");
    </script>
  </body>
</html>
"""

PIPELINE: Optional[SOPRagPipeline] = None


class ChatHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            data = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404, "Not Found")
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON body."})
            return

        message = str(payload.get("message", "")).strip()
        if not message:
            self._send_json(400, {"error": "Message is required."})
            return

        if PIPELINE is None:
            self._send_json(500, {"error": "Pipeline not initialized."})
            return

        result = PIPELINE.answer_question(message)
        self._send_json(200, result)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality SOP Chat UI")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--pdf-dir", type=str, default="")
    parser.add_argument("--vector-db", type=str, default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    pdf_dir = Path(args.pdf_dir) if args.pdf_dir else base_dir / "pdfs"
    vector_db = Path(args.vector_db) if args.vector_db else base_dir / "faiss_index"

    global PIPELINE
    PIPELINE = SOPRagPipeline(
        pdf_dir=str(pdf_dir),
        vector_db_path=str(vector_db),
    )

    if args.index:
        PIPELINE.load_and_process_documents()
    else:
        PIPELINE.load_vector_store()

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    print(f"Chat UI running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
