# ------------------------------------------------------------------------------
# File: llm.py
# Description: llm tools for KriRAG. Relies on a openai-comptaible api, default through llama.cpp server docker container.
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

import json
import re

import requests

question_and_reason_prompt = {
    "en": "You are an AI assisting a criminal investigation, analyzing case files for knowledge discoveries. You follow strict logical and deductive reasoning, and will only present information for which you have a complete overview of. Do not make assumptions, or add any superfluous information. {extra}You receive a new document with ID {doc_id}: '{text}'. Investigate document {doc_id} grounded in the QUERY: '{query}'. Generate a JSON object with 1) questions: a list of investigative questions (based on e.g., objects, actions, events, entities) that are directly related to the QUERY in {doc_id}. 2) reason: discuss whether document {doc_id} answers the QUERY. 3) score: if the document is 0 irrelevant, 1 somewhat relevant, 2 relevant, or 3 extremely relevant. 4) a summary of vital details uncovered in {doc_id}.",
}
memory_prompt = "You are an AI assisting a criminal investigation, analyzing case files. You follow abductive reasoning and logic. Do not make assumptions, or add any superfluous information. From the following data:\n{previous_information}, create a summary of vital information related to the query: '{query}'. Make sure to reference the ID '{DOC_ID}' for your findings, and keep all previous document references."


headers = {"Content-Type": "application/json"}

schema = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
        "reason": {"type": "string"},
        "score": {"type": "integer", "enum": [0, 1, 2, 3]},
        "summary": {"type": "string"},
    },
    "required": ["questions", "reason", "score", "summary"],
}

schema_summ = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
    },
    "required": ["summary"],
}

schema_findings = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
        },
        "references": {
            "type": "array",
            "items": {
                "type": "string",
            },
        },
    },
    "required": ["summary", "references"],
}


schemas = {
    "default": schema,
    "summary": schema_summ,
    "findings": schema_findings,
}


def pred(
    instruction,
    ip_address: str,
    port: int,
    max_tokens=1000,
    use_schema: str = "default",
    temp=0.0,  # temperature. 0: deterministic, 1+: random
    # min_p=0.1,  # minimum probability
    # max_p=0.9,  # maximum probability
    # top_p=0.9,  # nucleus sampling
    # top_k=40,  # consider top k tokens at each generation step
    evaluate: bool = False,  # apply eval
):
    url = f"http://{ip_address}:{port}/completion"  # llama.cpp server
    if len(instruction) == 0:
        raise ValueError("Instruction cannot be empty")

    data = {
        "prompt": instruction,
        "n_predict": max_tokens,
        "temperature": temp,
        "repeat_penalty": 1.2,  # 1.1 default,
    }
    if use_schema:
        data["json_schema"] = schemas[use_schema]

    response = requests.post(url, headers=headers, data=json.dumps(data)).json()
    response = response["content"]
    if evaluate:
        return parse_llm_output(response)
    return response


def parse_llm_output(response: str):
    if not response:
        return response
    # spacing!
    response = response.replace("\n", " ")
    response = response.replace("\t", " ")
    response = re.sub(r"\s+", " ", response)
    response = response.strip()
    # markdown ticks
    response = response.replace("```python", "")
    response = response.replace("```json", "")
    response = response.replace("```", "")

    response = response.replace("false", "False")
    response = response.replace("true", "True")
    response = response.replace("null", "None")

    obj = eval(response)
    if isinstance(obj, dict):
        # unify keys in case of capitalization.
        obj = {k.lower(): v for k, v in obj.items()}
    return obj


def ask_llm(
    query: str,
    text: str,
    ip_address: str,
    port: int,
    extra: str = "",
    doc_id: str = "ID",
    temp: float = 0.0,
    tokens: int = 150,
    prompt_source: dict = None,  # see "question_and_reason_prompt" above.
    lang: str = "en",
    verbose: bool = False,
) -> dict:
    text = re.sub(r"\.{3,}", "...", text)

    if extra:
        extra = f"You have info from previous interrogations: '{extra}'. Use this info to guide your reasoning if relevant."

    instruction = prompt_source[lang].format(
        query=query,
        text=text,
        extra=extra,
        doc_id=doc_id,
    )
    if verbose:
        print("Instruction", instruction)

    output = pred(
        instruction=instruction,
        ip_address=ip_address,
        port=port,
        temp=temp,
        max_tokens=tokens,
        use_schema="default",
    )
    if verbose:
        print("-*-" * 40)
        print(output)
        print("-*-" * 40)
    try:
        output = parse_llm_output(output)
    except SyntaxError as e:
        print("SyntaxError. Returning raw output")
    return output
