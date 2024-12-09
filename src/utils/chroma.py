# ------------------------------------------------------------------------------
# File: chroma.py
# Description: wrapper around ChromaDB for offline usage in KriRAG
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

import logging
import os
from typing import List, Tuple

import chromadb
from chromadb import Client, Collection, Documents, EmbeddingFunction, Embeddings
from chromadb.config import Settings
from chromadb.utils.batch_utils import create_batches
from pandas import DataFrame
from sentence_transformers import SentenceTransformer


class CustomEmbedder(EmbeddingFunction):
    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(
            input, convert_to_numpy=True, batch_size=self.batch_size
        )
        return [e.tolist() for e in embeddings]


def get_collection(collection_name: str = "rag"):
    chroma_client = chromadb.Client()
    collection = chroma_client.get_collection(
        name=collection_name,
    )
    return collection


def get_client(
    persist: bool = True,
    delete: bool = False,
    embedding_model: SentenceTransformer = None,
    collection_name: str = "rag",
) -> Tuple[chromadb.Client, chromadb.Collection]:
    _settings = Settings(anonymized_telemetry=False)
    if persist:
        chroma_client = chromadb.PersistentClient(settings=_settings)
    else:
        chroma_client = chromadb.Client(settings=_settings)

    if delete:
        try:
            logging.info(f"Deleting collection {collection_name}")
            chroma_client.delete_collection(name=collection_name)
        except Exception as e:
            logging.error(e)
            logging.info("Proceeding as normal.")

    embedding_function = CustomEmbedder(embedding_model)
    collection = chroma_client.create_collection(
        name=collection_name, embedding_function=embedding_function, get_or_create=True
    )
    logging.info(f"Init collection {collection_name}")
    return chroma_client, collection


def peek(collection: chromadb.Collection, count: int):
    sample = collection.peek(limit=count)
    for key, val in sample.items():
        if not val:
            continue
        print(key)
        val = val[0]
        if isinstance(val, list):
            print("\t", val[:10], "...")
        else:
            print("\t", val)


def update_collection(
    client: Client,
    model: SentenceTransformer,
    df: DataFrame,
    collection: Collection,
    DOCUMENTS_TEXT_COLUMN,
    DOCUMENTS_ID_COLUMN,
):
    if collection.count() > 0:
        print(f"Collection has {collection.count()} items. Safe to continue :)")
    else:
        print("Collection is empty. Adding data...")
        document_meta = []
        for i, row in df.iterrows():
            document_meta.append(
                {
                    "id": row[DOCUMENTS_ID_COLUMN],
                }
            )

        documents = df[DOCUMENTS_TEXT_COLUMN].tolist()
        embeddings = model.encode(documents, show_progress_bar=True).tolist()
        ids = df.index.map(str).tolist()

        batches = create_batches(
            api=client,
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=document_meta,
        )

        for batch in batches:
            collection.add(
                ids=batch[0],
                embeddings=batch[1],
                metadatas=batch[2],
                documents=batch[3],
            )


def get_matching_documents(
    collection: Collection, query: str, n_results: int
) -> List[str]:
    query_result = collection.query(query_texts=query, n_results=n_results)
    query_result = {
        k: v[0] for k, v in query_result.items() if isinstance(v, list) and len(v) > 0
    }
    metadata = query_result["metadatas"]
    documents = [d["document"] for d in metadata]
    documents = sorted(list(set(documents)))
    return documents
