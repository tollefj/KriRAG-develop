import json
import os
import re
from typing import Any, Dict, List, Tuple

import pandas as pd

from llm import pred

MIN_TOKENS: int = 100  # min summary tokens
MAX_TOKENS: int = 2000  # output tokens

processing_prompt: str = (
    "You are an AI assisting a criminal investigation, analyzing case files. You follow strict logical and deductive reasoning, and will only present information for which you have a complete overview of. Avoid assumptions and uncertainty. Do not repeat yourself. You receive the following information: '{text}'. Assess the relevance of each document to the query '{query}' and write a highly detailed summary (including involved persons, objects, locations and other entities), based on the most relevant documents. Return a JSON object with the summary and references to the most relevant documents."
)


def process_case(case_jsonl_path: str, ip_address: str, port: int) -> Tuple[str, Dict[str, Any]]:
    doc_findings: List[Dict[str, Any]] = []
    with open(case_jsonl_path, "r", encoding="utf-8") as f:
        doc_findings = [json.loads(x) for x in f.readlines()]
    print(f"Found {len(doc_findings)} answers.")
    doc_df = pd.DataFrame(doc_findings)
    doc_df = pd.concat(
        [
            doc_df.drop(["llm_output"], axis=1),
            doc_df["llm_output"].apply(pd.Series),
        ],
        axis=1,
    )
    query: str = doc_df["query"][0]
    print(f"Processing documents for query: {query}")
    summaries: List[str] = doc_df["summary"].unique().tolist()
    text: str = "\n".join(summaries)
    instruction: str = processing_prompt.format(text=text, query=query)
    output: Dict[str, Any] = pred(
        instruction,
        ip_address=ip_address,
        port=port,
        use_schema="findings", max_tokens=MAX_TOKENS, evaluate=True
    )
    return query, output


def meta_summary(case_path: str, ip_address: str, port: int) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []
    for f in os.listdir(case_path):
        full_path: str = os.path.join(case_path, f)
        query, processed = process_case(full_path, ip_address=ip_address, port=port)
        if "summary" in processed:
            metas.append(
                {
                    "query": query,
                    "summary": processed["summary"],
                }
            )
    return metas
