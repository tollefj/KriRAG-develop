# ------------------------------------------------------------------------------
# File: rag.py
# Description: rag pipeline for KriRAG
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------
import os
import re
from datetime import datetime
from typing import List

import jsonlines
import streamlit as st
from chromadb.types import Collection

from llm import (
    ask_llm,
    memory_prompt,
    parse_llm_output,
    pred,
    question_and_reason_prompt,
)
from utils.batch import get_sentence_batches
from utils.chroma import get_matching_documents


def run_rag(
    queries: List[str],
    collection: Collection,
    ip_address: str,
    port: int,
    lang: str = "en",
    top_n: int = 10,
    llm_ctx_len: int = 8168,
    new_tokens: int = 2048,
) -> str:
    # print all locals that rag is running with:
    print(locals())

    start_of_program: str = datetime.now().strftime("%Y%m%d-%H%M%S")
    _metadata: List[dict] = collection.get()["metadatas"]
    _documents: set = set([d["document"] for d in _metadata])
    if top_n == -1:
        top_n = len(_documents)

    case_folder = f"RAG_Top{top_n}_{start_of_program}"
    rag_path = os.path.join("output", case_folder)

    for query in queries:
        st.write(f"Processing query: {query}")
        print(f"Query: {query}")
        documents = get_matching_documents(
            collection=collection,
            query=query,
            n_results=top_n,
        )
        print(f"Reduced from {top_n} to {len(documents)} documents")

        # 1 word can easily span 3-4 tokens
        # thus, to match the context length: ctx_len//4
        TOKEN_LEN: int = llm_ctx_len // 4

        timestamp: str = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename: str = re.sub(r"[^\w\s]", "", query)
        filename = filename.replace(" ", "-")

        output_path = os.path.join(rag_path, f"{timestamp}_{filename}.jsonl")
        os.makedirs(os.path.join("output", case_folder), exist_ok=True)

        # hold the "summary" field across documents to feed to the LLM
        QUERY_MEMORY: List[str] = []

        with jsonlines.open(output_path, "w") as writer:
            progress_text = f"Processing {len(documents)} documents..."
            progress_bar = st.progress(0, text=progress_text)

            for i, matched_doc in enumerate(documents):
                current_percentage = (i + 1) / len(documents)
                progress_bar.progress(
                    current_percentage, text=f"Document {i + 1}/{len(documents)}"
                )
                matching_full_docs = collection.get(
                    where={"document": {"$in": [matched_doc]}},
                )
                texts: List[str] = matching_full_docs["documents"]
                print(f"Doc {matched_doc} has {len(texts)} sentences")

                DOC_ID: str = matched_doc
                print(f"Doc ID: {DOC_ID}")

                # - some documents are LONG, batch them into smaller chunks
                batches = get_sentence_batches(texts, TOKEN_LEN)
                sentence_batch_map = batches["map"]
                batches = batches["batches"]

                for batch, batch_texts in batches.items():
                    print(f"Working with batch {batch + 1}/{len(batches)}")
                    full_text: str = " ".join(batch_texts)
                    prev_info: str = ""
                    # no memory for the first batch
                    if len(QUERY_MEMORY) > 0:
                        prev_info = pred(
                            instruction=memory_prompt.format(
                                previous_information=QUERY_MEMORY,
                                query=query,
                                DOC_ID=DOC_ID,
                            ),
                            ip_address=ip_address,
                            port=port,
                            max_tokens=1000,
                            use_schema="summary",
                        )
                        prev_info = parse_llm_output(prev_info)
                        print(f"Identified previous information: {prev_info}")
                        if isinstance(prev_info, dict) and "summary" in prev_info:
                            prev_info = prev_info["summary"]

                    print(f"Getting preds from LLM with previous info: {prev_info}")

                    llm_output = ask_llm(
                        query=query,
                        text=full_text,
                        ip_address=ip_address,
                        port=port,
                        extra=prev_info,
                        doc_id=DOC_ID,
                        tokens=new_tokens,
                        prompt_source=question_and_reason_prompt,
                        verbose=False,
                        lang=lang,
                    )

                    tmp_summary: str = ""
                    if isinstance(llm_output, dict) and "summary" in llm_output:
                        tmp_summary = llm_output["summary"]
                    QUERY_MEMORY = [prev_info, tmp_summary]
                    print(f"Updated previous info:", QUERY_MEMORY)
                    json_record = {
                        "id": matched_doc,
                        "batch": batch,
                        "query": query,
                        "llm_output": llm_output,
                        "sentences_in_batch": sentence_batch_map[batch],
                        "text": full_text,
                        "memory": QUERY_MEMORY,
                    }

                    try:
                        keys = llm_output.keys()
                        assert "questions" in keys
                        assert "score" in keys
                        assert "summary" in keys
                    except AssertionError:
                        print("Error: missing keys in llm_output")
                        print(llm_output)
                        print("___")
                        llm_output = {
                            "questions": ["Error generating questions..."],
                            "score": -1,
                            "summary": "",
                        }
                        continue

                    with st.expander(
                        f"Results for {matched_doc} (relevance score: {llm_output['score']}/3)"
                    ):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"#### Query\n{query}")
                            st.markdown(f"#### Generated questions")
                            for q in llm_output["questions"]:
                                if "question" in q:
                                    st.markdown(f"- {q['question']}")
                            st.markdown(f"#### Summary\n{llm_output['summary']}")

                            st.markdown(f"#### Memory")
                            for _prev_info in QUERY_MEMORY:
                                if len(_prev_info) > 3:
                                    st.markdown(f"- {_prev_info}")

                        with col2:
                            st.markdown(f"#### Full text")
                            st.write(full_text)
                        st.divider()
                    writer.write(json_record)
    return rag_path
