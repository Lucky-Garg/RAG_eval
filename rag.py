"""
Basic RAG pipeline:  Load -> Chunk -> Embed -> Store (Chroma) -> Retrieve -> Generate (Groq)
"""
import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"   # local, free

SYSTEM_PROMPT = (
    "You are a helpful HR assistant. Answer the question using ONLY the context provided. "
    "If the context does not contain the answer, reply exactly: "
    "'I don't have enough information to answer that.' Be concise."
)


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> list[str]:
    """Word-based sliding-window chunking with overlap."""
    words = text.split()
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


class RAGPipeline:
    def __init__(self, collection_name: str = "acme_policies", top_k: int = 3):
        self.top_k = top_k
        self.llm_model = os.getenv("LLM_MODEL")
        self.llm = OpenAI(
            api_key=os.getenv("LLM_API_KEY", "ollama"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        # cosine distance works best for sentence-transformer embeddings
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, folder: str = "data") -> int:
        """Read every .txt file, chunk it, and store the chunks in the vector DB."""
        ids, docs, metas = [], [], []
        for path in Path(folder).glob("*.txt"):
            for i, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
                ids.append(f"{path.stem}-{i}")
                docs.append(chunk)
                metas.append({"source": path.name, "chunk": i})
        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
        return len(ids)

    def retrieve(self, query: str) -> list[str]:
        res = self.collection.query(query_texts=[query], n_results=self.top_k)
        return res["documents"][0]

    def generate(self, query: str, contexts: list[str]) -> str:
        context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
        resp = self.llm.chat.completions.create(
            model=self.llm_model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
            ],
        )
        return resp.choices[0].message.content.strip()

    def query(self, question: str) -> tuple[str, list[str]]:
        """Returns (answer, retrieved_contexts) - both are needed for evaluation."""
        contexts = self.retrieve(question)
        return self.generate(question, contexts), contexts


if __name__ == "__main__":
    rag = RAGPipeline()
    print(f"Ingested {rag.ingest()} chunks")
    while True:
        q = input("\nAsk (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        answer, ctx = rag.query(q)
        print(f"\nAnswer: {answer}\n\nRetrieved chunks: {len(ctx)}")