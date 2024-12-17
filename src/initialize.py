import os
import shutil
import zipfile
from typing import Dict, List, Tuple

import nltk
import pandas as pd
import streamlit as st
from chromadb import PersistentClient
from chromadb.types import Collection
from chromadb.utils.batch_utils import create_batches
from sentence_transformers import SentenceTransformer

from utils.chroma import get_client

EMBEDDING_MODEL = "sbert"
LANG = "english"

valid_exts = [".txt", ".json", ".jsonl"]

with st.spinner("Loading SentenceTransformer model..."):
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL,
        backend="openvino",  # we optimize cpu-inference to reduce docker container image (w/ cuda drivers etc.)
        local_files_only=True,
    )


def load_txt_from_folder(folder_path: str, lang: str = LANG) -> pd.DataFrame:
    # walk all files in the directory!
    parsed_data = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    _doc = f.readlines()
                    filename_no_ext = os.path.splitext(file)[0]
                    parsed_data.extend(parse_document(_doc, lang, document_name=filename_no_ext))

    return pd.DataFrame(parsed_data)


def parse_document(
    docs: List[str],
    lang: str = LANG,
    strip_newlines: bool = True,
    document_name: str = "",
) -> List[str]:
    if strip_newlines:
        print(f"Stripping newlines from {len(docs)} documents.")
        docs = [d.replace("\n", "") for d in docs]

    def sentencize(text):
        return nltk.sent_tokenize(text, language=lang)

    parsed_data = []
    sentences = []
    paragraph_sent_map = []
    for d_id, paragraph in enumerate(docs):
        d_sents = sentencize(paragraph)
        paragraph_sent_map.extend([d_id] * len(d_sents))
        sentences.extend(d_sents)

    for s_id, sent in enumerate(sentences):
        _d_id = paragraph_sent_map[s_id]
        parsed_data.append(
            {
                "id": document_name,
                "page_id": _d_id,
                "sent_id": s_id,
                "text": sent,
            }
        )

    return parsed_data


@st.cache_data
def load_and_cache_documents(uploaded_file, lang: str):
    print(f"Loading: {uploaded_file.name} with language: {lang}")
    ext = uploaded_file.name.split(".")[-1] if uploaded_file else None
    filename = uploaded_file.name.split(".")[0] if uploaded_file else None
    df = pd.DataFrame()
    if ext == "txt":
        uploaded_file = uploaded_file.read().decode("utf-8")
        print(f"Loaded single document with {len(uploaded_file)} paragraphs.")

        df = pd.DataFrame(parse_document(uploaded_file, lang=lang, document_name=filename))

    elif ext == "zip":
        with zipfile.ZipFile(uploaded_file, "r") as z:
            # remove the temp folder if it exists
            if os.path.exists("temp"):
                shutil.rmtree("temp", ignore_errors=True)
            z.extractall("temp")
            df = load_txt_from_folder("temp")

    if df.empty:
        raise ValueError("No data found in the uploaded file.")

    print(f"Loaded {len(df)} documents.")
    # columns: id/page_id/sent_id/text
    num_docs = df["id"].nunique()
    num_pages = df["page_id"].nunique()
    num_sents = df.shape[0]
    st.info(f"Found {num_docs} documents, {num_pages} paragraphs, and {num_sents} sentences")
    return {
        "data": df.to_dict(orient="records"),
        "num_docs": num_docs,
        "num_pages": num_pages,
        "num_sents": num_sents,
    }


def populate_collection(
    data: List[Dict[str, any]],
    collection_name: str,
    delete=False,
    BATCH_SIZE=32,
) -> Tuple[PersistentClient, Collection]:
    client, collection = get_client(
        persist=True,  # persist: store to disk (under the `chroma` folder)
        delete=delete,  # WARNING: enable ONLY if doing changes to the data
        embedding_model=embedding_model,
        collection_name=collection_name,
    )
    if collection.count() == 0:
        document_meta = []
        meta_text = "Adding metadata..."
        meta_bar = st.progress(0, text=meta_text)
        for percent_complete, row in enumerate(data):
            print(f"Adding metadata for document: {row['id']}")
            document_meta.append(
                {
                    "document": row["id"],
                    "sent_id": row["sent_id"],
                    "page_id": row["page_id"],
                }
            )
            current_perc = (percent_complete + 1) / len(data)
            current_perc = min(current_perc, 1.0)
            meta_bar.progress(current_perc)
        meta_bar.empty()

        documents = [d["text"] for d in data]

        with st.spinner("Computing embeddings..."):
            embeddings = embedding_model.encode(
                documents,
                show_progress_bar=True,
                batch_size=BATCH_SIZE,
            ).tolist()

        # a reference key for each document
        ids = []
        for d_id, d in enumerate(data):
            ids.append(f"{d_id}-{d['id']}-{d['page_id']}-{d['sent_id']}")

        batches = create_batches(
            api=client,
            ids=ids,
            embeddings=embeddings,
            metadatas=document_meta,
            documents=documents,
        )

        # i: index, e: embedding, m: metadata, d: document
        for i, e, m, d in batches:
            collection.add(i, e, m, d)

    return client, collection
