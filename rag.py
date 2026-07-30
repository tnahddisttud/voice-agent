import sys
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

from config import settings


def retrieve(
    client: QdrantClient,
    embedder: SentenceTransformer,
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Embed the query and return the top-k most similar chunks.
    """
    query_vector = embedder.encode(query).tolist()

    hits = client.query_points(
        collection_name=settings.collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    return [{**hit.payload, "score": round(hit.score, 4)} for hit in hits.points]


def build_context(retrieved_chunks: list[dict]) -> str:
    """Build a context string from the retrieved chunks."""
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        # Using chunk_text to provide the breadcrumb hierarchy and the content
        parts.append(f"[Source {i}]\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)


def rag(query: str, top_k: int = 5, client=None, embedder=None):
    """
    End-to-end RAG pipeline:
      1. Retrieve relevant chunks from Qdrant
      2. Format them as a context block
      3. Send context + query to Groq and return the answer
    """
    if embedder is None:
        embedder = SentenceTransformer(settings.embedding_model)
    if client is None:
        client = QdrantClient(path=settings.qdrant_path)

    # Step 1 — Retrieve
    chunks = retrieve(client, embedder, query, top_k=top_k)
    if not chunks:
        return "No relevant content found in the document.", ""

    # Step 2 — Build context
    context = build_context(chunks)

    # Step 3 — Generate
    SYSTEM_PROMPT = """You are a helpful HR assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information. Respond with only single sentence."""

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    llm = ChatGroq(model=settings.groq_model, temperature=0.2)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", user_message),
    ]

    response = llm.invoke(messages)
    return response.content, context


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "How many casual leaves am I entitled to?"

    print(f"Question: {query}\n")
    print("Generating answer...\n")

    answer, context = rag(query)

    print(f"ANSWER:\n{answer}")
    print(f"\n{60*'='}")
    print(f"\nSOURCES:\n{context}")


if __name__ == "__main__":
    main()
