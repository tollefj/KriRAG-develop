# ------------------------------------------------------------------------------
# File: data_util.py
# Description: utils for data loaders
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

from collections import defaultdict
from typing import List

import jsonlines
import nltk
import pandas as pd


def sentencize(text, lang="norwegian"):
    return nltk.sent_tokenize(text, language=lang)


example_paths = {
    "lovdata": "lovdata-20-03-23.jsonl",
    "lovdata-subset": "lovdata-20-03-23.subset.jsonl",
}


def txt_loader(path: str) -> pd.DataFrame:
    sentences = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if len(line.strip()) > 0:
                sentences.extend(sentencize(line))

    df = pd.DataFrame(sentences, columns=["text"])
    df.head()


def jsonl_loader_sentences(
    path: str,
    id_key="id",
    sentence_key="sent_text",
    output_columns=List[str],
) -> pd.DataFrame:
    data = defaultdict(list)
    with jsonlines.open(path) as reader:
        for obj in reader:
            document = obj[id_key]
            sentences = obj[sentence_key]
            data[document] = sentences

    df = pd.DataFrame(data.items(), columns=output_columns)
    # df = df.explode(sentence_key)
    df = df.drop_duplicates(subset=sentence_key)
    df = df.reset_index(drop=True)

    return df


def jsonl_loader(
    path: str,
    id_key="id",
    paragraph_key="paragraphs",
    output_columns=List[str],
) -> pd.DataFrame:
    data = defaultdict(list)
    with jsonlines.open(path) as reader:
        for obj in reader:
            document = obj[id_key]
            document = document.split("/")[-1]
            paragraphs = obj[paragraph_key]
            sentences = []
            for paragraph in paragraphs:
                sentences.extend(sentencize(paragraph))
            data[document] = sentences

    df = pd.DataFrame(data.items(), columns=output_columns)
    df = df.explode("text")
    df = df.drop_duplicates(subset="text")
    df = df.reset_index(drop=True)

    return df
