"""
Career chat RAG + Gradio — deploy entrypoint (e.g. Hugging Face Spaces).

Loads `vector_db/` when present; otherwise ingests from `me/`.
Set FORCE_REBUILD_INDEX=1 to re-embed from `me/` even if `vector_db/` exists.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownTextSplitter
from openai import OpenAI
from pypdf import PdfReader

# Resolve paths relative to this file (works from repo root or HF Space).
_ROOT = Path(__file__).resolve().parent
os.chdir(_ROOT)

load_dotenv(override=True)

# --- Config (env overrides) ---
YOUR_NAME = os.getenv("YOUR_NAME", "Sam")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

KNOWLEDGE_DIR = "me"
DB_NAME = "vector_db"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5


def load_raw_texts() -> list[str]:
    texts: list[str] = []
    for path in glob.glob(os.path.join(KNOWLEDGE_DIR, "*")):
        if path.endswith(".pdf"):
            reader = PdfReader(path)
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            texts.append("\n".join(parts))
        elif path.endswith(".txt") or path.endswith(".md"):
            with open(path, encoding="utf-8") as f:
                texts.append(f.read())
        else:
            print(f"Skipping (not pdf/txt/md): {path}")
    return texts


def chunk_documents(texts: list[str]) -> list[Document]:
    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    docs = [Document(page_content=t) for t in texts if t.strip()]
    return splitter.split_documents(docs)


def build_vector_store(chunks: list[Document]) -> Chroma:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )
    n = store._collection.count()
    print(f"Stored {n} chunks in {DB_NAME}/")
    return store


def _force_rebuild() -> bool:
    return os.getenv("FORCE_REBUILD_INDEX", "").lower() in ("1", "true", "yes")


def create_retriever() -> VectorStoreRetriever:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    raw = load_raw_texts()

    def build_from_raw() -> VectorStoreRetriever:
        if not raw:
            raise RuntimeError(
                f"No ingestible files in ./{KNOWLEDGE_DIR}. Add summary.txt and/or PDFs, or ship a populated {DB_NAME}/."
            )
        pieces = chunk_documents(raw)
        store = build_vector_store(pieces)
        return store.as_retriever(search_kwargs={"k": TOP_K})

    if _force_rebuild():
        return build_from_raw()

    if os.path.exists(DB_NAME):
        store = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
        if store._collection.count() > 0:
            return store.as_retriever(search_kwargs={"k": TOP_K})
        if raw:
            return build_from_raw()
        raise RuntimeError(
            f"{DB_NAME}/ exists but is empty. Add sources under ./{KNOWLEDGE_DIR}/ or remove {DB_NAME}/."
        )

    if raw:
        return build_from_raw()

    raise RuntimeError(
        f"No {DB_NAME}/ and no ingestible files under ./{KNOWLEDGE_DIR}/. "
        "Add knowledge files or commit a built vector index."
    )


def normalize_history(history):
    if not history:
        return []
    out = []
    for msg in history:
        if isinstance(msg, dict):
            content = msg.get("content")
            if content is not None and not isinstance(content, str):
                content = str(content) if content else ""
            out.append({"role": msg["role"], "content": content or ""})
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            u, a = msg[0], msg[1]
            out.append({"role": "user", "content": u if isinstance(u, str) else str(u)})
            out.append({"role": "assistant", "content": a if isinstance(a, str) else str(a)})
    return out


def record_user_details(email: str, name: str = "Name not provided", notes: str = "not provided"):
    path = Path("me") / "leads.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{name} | {email} | {notes}\n")
    return {"recorded": "ok"}


def record_unknown_question(question: str):
    path = Path("me") / "unknown_questions.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{question}\n")
    return {"recorded": "ok"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "record_user_details",
            "description": "Record that the user wants to stay in touch and gave an email.",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "User email"},
                    "name": {"type": "string", "description": "User name if provided"},
                    "notes": {"type": "string", "description": "Extra context"},
                },
                "required": ["email"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_unknown_question",
            "description": "Record a question you could not answer from the provided context.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    },
]


class CareerBot:
    def __init__(self, retriever: VectorStoreRetriever):
        self.client = OpenAI()
        self.model = CHAT_MODEL
        self._retriever = retriever

    def _system_prompt(self, context: str) -> str:
        return f"""You are {YOUR_NAME}, chatting on your personal site about your career, skills, and background.
Use only the context below when stating facts. If something is not covered, say you do not have that information
and offer to connect by email. Be professional and friendly.

If the user wants to stay in touch, ask for their email and call record_user_details.
If you cannot answer from the context, call record_unknown_question with their question.

## Context about {YOUR_NAME}:
{context}
"""

    def _handle_tools(self, tool_calls):
        results = []
        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            fn = globals().get(name)
            payload = fn(**args) if callable(fn) else {}
            results.append(
                {"role": "tool", "content": json.dumps(payload), "tool_call_id": call.id}
            )
        return results

    def chat(self, message, history):
        docs = self._retriever.invoke(message)
        context = "\n\n".join(d.page_content for d in docs)
        messages = (
            [{"role": "system", "content": self._system_prompt(context)}]
            + normalize_history(history)
            + [{"role": "user", "content": message}]
        )
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOLS,
            )
            choice = response.choices[0]
            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)
                messages.extend(self._handle_tools(choice.message.tool_calls))
            else:
                return choice.message.content or ""


def main() -> None:
    retriever = create_retriever()
    bot = CareerBot(retriever)
    port = int(os.environ.get("PORT", "7860"))
    demo = gr.ChatInterface(
        bot.chat,
        title=f"{YOUR_NAME} — Career chat",
    )
    demo.launch(server_name="0.0.0.0", server_port=port)


if __name__ == "__main__":
    main()
