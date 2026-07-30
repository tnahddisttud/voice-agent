import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker import HierarchicalChunker
from hierarchical.postprocessor import ResultPostprocessor

from config import settings


def load_document(source: str):
    """
    Parse a document using Docling.
    Returns a DoclingDocument object.
    """
    converter = DocumentConverter()
    result = converter.convert(source)
    ResultPostprocessor(result).process()
    return result.document


def convert_chunk(doc_chunk) -> dict:
    """
    Convert a Docling DocChunk into a plain dict.
    headings   → list preserved as-is
    content    → paragraph text
    chunk_text → breadcrumb + content  (what gets embedded)
    """
    headings   = doc_chunk.meta.headings or []
    content    = doc_chunk.text.strip()
    breadcrumb = " > ".join(headings)
    chunk_text = f"{breadcrumb}\n\n{content}" if breadcrumb else content

    return {
        "headings":   headings,
        "content":    content,
        "chunk_text": chunk_text,
    }


def main() -> None:
    print(f"Loading document from {settings.github_raw_url} …")
    doc = load_document(settings.github_raw_url)
    print(f"Loaded document: {doc.name}")

    print("Chunking …")
    chunker = HierarchicalChunker()
    doc_chunks = list(chunker.chunk(doc))
    chunks = [convert_chunk(c) for c in doc_chunks]
    print(f"Total chunks: {len(chunks)}")

    print(f"Loading embedding model '{settings.embedding_model}' …")
    embedder = SentenceTransformer(settings.embedding_model)

    chunk_texts = [c["chunk_text"] for c in chunks]
    print(f"Embedding {len(chunk_texts)} chunks …")
    embeddings = embedder.encode(chunk_texts, show_progress_bar=True)
    print(f"Embedding shape: {embeddings.shape}")

    print(f"Opening Qdrant at '{settings.qdrant_path}' …")
    client = QdrantClient(path=settings.qdrant_path)

    dim = embedder.get_sentence_embedding_dimension()
    client.recreate_collection(
        collection_name=settings.collection_name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
    print(f"Collection '{settings.collection_name}' created.")

    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={
                "headings":   chunk["headings"],
                "content":    chunk["content"],
                "chunk_text": chunk["chunk_text"],
            },
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    result = client.upsert(collection_name=settings.collection_name, points=points, wait=True)
    print(f"Indexed {len(points)} points — status: {result.status}")

    info = client.get_collection(settings.collection_name)
    print(f"Points     : {info.points_count}")
    print(f"Dimensions : {info.config.params.vectors.size}")

    # Important: release the file lock so agent.py can open the same path
    client.close()
    print("\n✅ Ingestion complete. Qdrant connection closed — safe to start agent.py now.")


if __name__ == "__main__":
    main()