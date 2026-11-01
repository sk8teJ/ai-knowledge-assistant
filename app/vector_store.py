
import hashlib
from pathlib import Path

import chromadb

from embeddings import get_embedding


# Путь к постоянной базе данных
DB_PATH = Path(__file__).resolve().parent.parent / ".chroma"

# Подключение к ChromaDB
client = chromadb.PersistentClient(path=str(DB_PATH))

# Коллекция документов
collection = client.get_or_create_collection(
    name="knowledge_documents",
    metadata={"hnsw:space": "cosine"}
)


def document_exists(document_id: str) -> bool:
    """
    Проверяет, существует ли документ в базе данных.
    """

    result = collection.get(
        where={"document_id": document_id},
        include=["metadatas"]
    )

    return len(result["ids"]) > 0


def add_chunks(
    chunks: list[str],
    source: str,
    document_id: str | None = None
) -> int:
    """
    Сохраняет фрагменты документа в ChromaDB.
    """

    if not chunks:
        return 0

    if document_id is None:
        document_id = source

    embeddings = [
        get_embedding(chunk)
        for chunk in chunks
    ]

    ids = []

    for index, chunk in enumerate(chunks):

        unique_text = f"{document_id}:{index}:{chunk}"

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    metadatas = [
        {
            "source": source,
            "document_id": document_id,
            "chunk_index": index
        }
        for index in range(len(chunks))
    ]

    # Удаляем старую версию этого документа
    collection.delete(
        where={"document_id": document_id}
    )

    # Сохраняем новую версию
    collection.upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


def search_chunks(
    query: str,
    n_results: int = 3
) -> list[dict]:
    """
    Находит наиболее подходящие фрагменты по вопросу.
    """

    if not query.strip():
        return []

    if collection.count() == 0:
        return []

    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    found_chunks = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        found_chunks.append({
            "text": document,
            "source": metadata["source"],
            "document_id": metadata.get("document_id"),
            "chunk_index": metadata.get("chunk_index"),
            "distance": distance
        })

    return found_chunks


def list_documents() -> list[dict]:
    """
    Возвращает список документов и количество их фрагментов.
    """

    if collection.count() == 0:
        return []

    result = collection.get(
        include=["metadatas"]
    )

    documents = {}

    for metadata in result["metadatas"]:

        document_id = metadata.get("document_id")
        source = metadata.get("source")

        if document_id not in documents:

            documents[document_id] = {
                "document_id": document_id,
                "source": source,
                "chunks_count": 0
            }

        documents[document_id]["chunks_count"] += 1

    return list(documents.values())


def delete_document(document_id: str) -> None:
    """
    Удаляет все фрагменты выбранного документа.
    """

    collection.delete(
        where={"document_id": document_id}
    )
