# ------------------------------------------------------------------------------
# File: ui.py
# Description: main ui for KriRAG (streamlit)
#
# License: Apache License 2.0
# For license details, refer to the LICENSE file in the project root.
#
# Contributors:
# - Tollef Jørgensen (Initial Development, 2024)
# ------------------------------------------------------------------------------

# keep these at the top
import dotenv
import streamlit as st

st.set_page_config(page_title="KriRAG", layout="wide")
dotenv.load_dotenv()

# remaining imports...
import json
import os
from datetime import datetime

import pandas as pd

from combine import meta_summary
from initialize import load_and_cache_documents, populate_collection
from rag import run_rag

default_queries = [
    "persons with residence and connections to the address (the crime scene) as owner, tenant, visitor, etc.",
    "how did the victim die (what is the cause of death?)",
    "details about the murder weapon (what is the murder weapon?)",
    "the victim's involvement in conflict or argument prior to death",
]
default_queries = [
    "personer med opphold og tilknytning til adressen (åstedet) som eier, leietaker, besøkende osv",
    "hvordan døde fornærmede (hva er dødsårsaken?)",
    "detaljer om drapsvåpenet (hva er drapsvåpenet?)",
    "fornærmedes (avdøde) involvering i konflikt eller krangel forut for døden",
]

# css hack to remove top header
st.markdown(
    """
    <style>
    [data-testid="stHeader"] {
    display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("KriRAG")
subtext = "query-based analysis for criminal investigations"
st.sidebar.header("Server Configuration")
ip_address = st.sidebar.text_input("LLM Docker Name or IP Address", value="krirag-api")
port = st.sidebar.number_input("API Port", value=8502, step=1)

# Add listeners for changes
if st.sidebar.button("Update Configuration"):
    st.session_state.ip_address = ip_address
    st.session_state.port = port
    st.success(f"Configuration updated to IP: {ip_address}, Port: {port}")

st.markdown(f"**{subtext}**")
st.divider()

query_area_exists = False
col1, col2 = st.columns(2)

initialization = {}

if "is_initialized" not in st.session_state:
    st.session_state.is_initialized = False
if "rag_started" not in st.session_state:
    st.session_state.rag_started = False
if "last_top_n" not in st.session_state:
    st.session_state["last_top_n"] = 1

with col1:
    st.write("### Data:")
    txt_upload = "Upload a single .txt file or a zip of multiple .txt files. The file names should have identifiable names for KriRAG to reference results."

    _uploaded = st.file_uploader(
        txt_upload, type=["txt", "zip"], accept_multiple_files=False
    )

    if _uploaded:
        initialization = load_and_cache_documents(_uploaded)

        with col2:
            st.markdown("### Queries:")
            query_area = st.text_area(
                "Your queries (one per line):",
                "\n\n".join(default_queries),
                height=400,
                key="queries",
            )

        print(initialization.keys())
        print(
            f"Initialization complete.\nData size: {initialization['num_pages']} pages, {initialization['num_sents']} sentences."
        )
        st.session_state.is_initialized = True

st.divider()

if st.session_state.is_initialized and "data" in initialization:
    to_delete = st.checkbox(
        "Delete previously computed data (local database)",
        value=False,
    )
    collection_name = (
        st.text_input(label="Name of your experiment:", value="KriRAG experiment 1")
        .replace(" ", "-")
        .lower()
    )
    queries = query_area.split("\n\n")
    queries = [q.strip() for q in queries if q]

    top_n = st.slider(
        "Number of candidate documents for each query.",
        1,
        min(initialization["num_pages"], 100),
        1,
    )
    if st.session_state.get("last_top_n") != top_n:
        st.session_state["last_top_n"] = top_n

    st.write(
        "Note: a higher slider value will increase processing time, but will likely find more relevant documents."
    )

    if st.button("Run KriRAG", disabled=st.session_state.rag_started):
        st.session_state.rag_started = True
        with st.spinner("Analyzing..."):
            _, collection = populate_collection(
                initialization["data"],
                collection_name=collection_name,
                delete=to_delete,
            )
            rag_path = run_rag(
                queries=queries,
                collection=collection,
                ip_address=ip_address,
                port=port,
                lang="en",
                top_n=top_n,
                llm_ctx_len=8168,
                new_tokens=4096,
            )
        with st.spinner("Processing findings..."):
            meta = meta_summary(rag_path, ip_address=ip_address, port=port)
            st.write("### Meta-summary of queries:")
            for m_id, meta_dict in enumerate(meta):
                st.write(f"Query: {meta_dict['query']}")
                st.write(f"{meta_dict['summary']}")
                st.divider()

        st.info(f"Analysis Complete! Download the CSV below.")
        st.session_state.rag_started = False

        all_data = []
        for file in sorted(os.listdir(rag_path)):
            if file.endswith(".jsonl"):
                file_path = os.path.join(rag_path, file)
                with open(file_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        data["id"] = file
                        all_data.append(data)
        results_df = pd.DataFrame(all_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(rag_path, "combined_results.csv")
        results_df.to_csv(csv_path, index=False)

        with open(csv_path, "rb") as f:
            st.download_button(
                label="Download Results",
                data=f,
                file_name=f"{collection_name.replace(' ', '_')}_{timestamp}.csv",
                mime="text/csv",
            )