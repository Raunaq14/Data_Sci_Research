# ==============================================================================
# InsightPod: A Conversational AI Analyst for FDA Device Data
#
# Final Build (v15 - Data Type Fix): This version resolves a data type
# mismatch bug in the year filter to ensure reliable analysis for all years.
# ==============================================================================

import os
import json
import re
from collections import Counter
import requests
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from thefuzz import process
import csv, time  # <-- add these

LOG_PATH = "chatbot_logs.csv"

def log_eval_row(row: dict):
    """Append one evaluation row to CSV (creates with header if missing)."""
    fieldnames = [
        "query_id", "query", "candidate_answer",
        "retrieved_ids", "cited_ids",
        "numeric_answer_log", "context_turn"
    ]
    safe = {k: row.get(k, "") for k in fieldnames}
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(safe)

# ========= Page Config & UI Theme =========
st.set_page_config(
    page_title="InsightPod",
    page_icon="💡",
    layout="centered",
    initial_sidebar_state="expanded",
)
st.title("💡 InsightPod")
st.caption("Your conversational AI analyst for FDA device data (2023-2024).")

# ========= Constants & Configuration =========

# --- Paths (IMPORTANT: Place CSVs in the same folder as this script) ---
CSV_2024 = "embedded_foi_meta_2024.csv"
CSV_2023 = "embedded_foi_meta_2023.csv"

# --- Qdrant Config ---
QDRANT_URL = os.environ.get("QDRANT_URL", "https://4a75c12d-b9a7-4a6c-a3b0-ad61bc2807d5.europe-west3-0.gcp.cloud.qdrant.io")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY","eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.cVE40MuYe7W6wWrVBFjF-P92AbCqoBVAhc_vxjCWt7o")
QDRANT_COLLECTION_NAME = "fda_reports"

# --- Ollama Config ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")
OLLAMA_TIMEOUT = 120

# ========= Caching for Performance =========

@st.cache_resource
def load_embedder():
    """Load the sentence transformer model once."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_metadata():
    """Load and merge CSV data once."""
    try:
        df24 = pd.read_csv(CSV_2024)
        df23 = pd.read_csv(CSV_2023)
        
        # NEW FIX: Reliably assign the year based on the source file.
        df24['YEAR'] = '2024'
        df23['YEAR'] = '2023'

    except FileNotFoundError as e:
        st.error(f"Fatal Error: Could not find data file '{e.filename}'. Please ensure both CSV files are in the same directory as the app.py script.")
        st.stop()

    for df in (df24, df23):
        if "summary_chunk" in df.columns:
            df.rename(columns={"summary_chunk": "text"}, inplace=True)
    
    df_all = pd.concat([df24, df23], ignore_index=True)
    df_all = df_all.dropna(subset=["text"]).reset_index(drop=True)
    
    # Ensure the YEAR column is consistently a string type
    df_all["YEAR"] = df_all["YEAR"].astype(str)

    return df_all

@st.cache_resource
def get_qdrant_client():
    """Get a Qdrant client instance."""
    if not QDRANT_API_KEY:
        st.error("Fatal Error: QDRANT_API_KEY environment variable not set. Please set it before running the app.")
        st.stop()
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

# ========= Load Resources on Startup =========
embedder = load_embedder()
df_base = load_metadata()
qdrant = get_qdrant_client()
ALLOWED_BRANDS = sorted({str(x) for x in df_base.get("BRAND_NAME", pd.Series([])).dropna().unique()})
ALLOWED_BRANDS += ["Dexcom G6", "Dexcom G7", "Libre 2", "Omnipod 5"]

# ========= Core Tools & Agent Logic =========

def run_analysis_tool(analysis_type: str, filter_by_device: str | None = None, filter_by_year: str | None = None) -> dict:
    """The data analysis tool. It runs a query against the local Pandas DataFrame."""
    if filter_by_year:
        filter_by_year = str(filter_by_year)

    if filter_by_year and filter_by_year not in ["2023", "2024"]:
        return {"error": f"I only have data for 2023 and 2024. I can't provide an analysis for the year {filter_by_year}."}

    dfc = df_base.copy()

    # Ensure the comparison is robust by comparing string to string.
    if filter_by_year:
        dfc = dfc[dfc["YEAR"] == filter_by_year]
    
    if filter_by_device:
        best_match, score = process.extractOne(filter_by_device, ALLOWED_BRANDS)
        if score > 80:
            dfc = dfc[dfc["BRAND_NAME"] == best_match]
            filter_by_device = best_match
        else:
            return {"error": f"Could not find a device matching '{filter_by_device}'."}

    if dfc.empty: return {"error": "No data found for the specified filters."}

    def _looks_like_failure(row):
        return "Malfunction" in str(row.get("issue_categories", ""))
    
    dfc_failures = dfc[dfc.apply(_looks_like_failure, axis=1)]
    if dfc_failures.empty: return {"error": "No failure-style reports found for the specified filters."}
    
    def get_examples(df_source, n=3):
        examples = []
        for text in df_source["text"].dropna().tolist():
            match = re.search(r'(?:event\s*)?description\s*:\s*(.+)', text, re.IGNORECASE)
            if match:
                description = match.group(1).strip()
                if len(description) > 20:
                    examples.append(description.split('.')[0])
            if len(examples) >= n:
                break
        return examples

    if analysis_type == "find_top_device_by_failure":
        by_brand = dfc_failures.groupby("BRAND_NAME").size().sort_values(ascending=False)
        top_device = by_brand.index[0]
        examples = get_examples(dfc_failures[dfc_failures["BRAND_NAME"] == top_device])
        return {
            "analysis_performed": "top_device_by_failure", "top_device": top_device,
            "count": int(by_brand.iloc[0]), "total_reports_in_scope": len(dfc_failures),
            "runners_up": {k: int(v) for k, v in by_brand.iloc[1:4].items()},
            "examples": examples
        }
    
    elif analysis_type == "find_top_cause_by_failure":
        df_causes = dfc_failures.copy()
        df_causes['issue_categories'] = df_causes['issue_categories'].apply(
            lambda x: re.split(r"[,\|\;]+", re.sub(r"[\[\]\'\"]", "", str(x)))
        )
        df_causes = df_causes.explode('issue_categories').dropna()
        df_causes['issue_categories'] = df_causes['issue_categories'].str.strip()
        df_causes = df_causes[df_causes['issue_categories'] != '']
        top_causes = df_causes['issue_categories'].value_counts()
        top_cause_name = top_causes.index[0]
        examples = get_examples(dfc_failures[dfc_failures['issue_categories'].str.contains(top_cause_name, case=False, na=False)])
        return {
            "analysis_performed": "top_cause_by_failure", "top_cause": top_cause_name,
            "count": int(top_causes.iloc[0]), "context_device": filter_by_device or "all devices",
            "context_year": filter_by_year or "2023-2024",
            "examples": examples
        }
    else:
        return {"error": f"Unknown analysis type requested: '{analysis_type}'"}

def search_for_examples(
    query: str,
    k: int = 25,
    year: str | None = None,
    return_meta: bool = False  # NEW: set True to also get ids/meta for evaluation
):
    """The RAG tool. It searches Qdrant for text snippets.

    Returns:
        - if return_meta=False (default): context_str
        - if return_meta=True: (context_str, retrieved_ids, metas)
            where:
              retrieved_ids = [str(hit.id), ...] in ranked order
              metas = [{"YEAR": ..., "BRAND_NAME": ..., "id": ...}, ...]
    """
    if year is not None:
        year = str(year).strip() or None  # empty -> None

    try:
        # Encode once; Qdrant expects float32
        vec = embedder.encode([query])[0].astype("float32")

        qfilter = None
        if year:
            qfilter = models.Filter(
                must=[models.FieldCondition(key="YEAR", match=models.MatchValue(value=year))]
            )

        hits = qdrant.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=vec,
            limit=int(k),
            query_filter=qfilter,
            with_payload=True
        )

        texts, ids, metas = [], [], []
        for h in hits:
            payload = h.payload or {}
            text = payload.get("text")
            if not text:
                continue
            texts.append(text)

            # robust id capture (stringify); fallback to a stable pseudo-id if needed
            hid = h.id
            if hid is None:
                # VERY rare; create a reproducible fallback id from payload
                hid = payload.get("REPORT_NUMBER") or payload.get("id") or hash(text)  # noqa: S324
            hid = str(hid)

            ids.append(hid)
            metas.append({
                "id": hid,
                "YEAR": payload.get("YEAR"),
                "BRAND_NAME": payload.get("BRAND_NAME"),
            })

        context = "\n\n---\n\n".join(texts)

        if return_meta:
            return context, ids, metas
        return context

    except Exception as e:
        st.error(f"Error searching Qdrant: {e}")
        return ("", [], []) if return_meta else ""
    
    

def get_analysis_type_from_query(query: str) -> str | None:
    """Step 1 of the new agent: A reliable keyword check for analysis type."""
    q_lower = query.lower()
    if any(k in q_lower for k in ["most failures", "most issues", "top device", "which device"]):
        return "find_top_device_by_failure"
    if any(k in q_lower for k in ["main cause", "common cause", "primary reason", "top issue"]):
        return "find_top_cause_by_failure"
    return None

def extract_filters_from_conversation(query: str, conversation_history: str) -> dict:
    """Step 2 of the new agent: A simple LLM call to extract filters with improved logic."""
    prompt = f"""Your task is to extract `device` and `year` filters from a new question based on the rules below.

**Analysis Rules:**

1.  **Check the New Question First:**
    - If the new question explicitly asks about "all devices", "every device", or "across devices", the `device` filter MUST be `null`.
    - If the new question explicitly names a device (e.g., "for the OMNIPOD 5 POD"), use that device name.
    - If the new question is asking "which device" or "what device", the `device` filter MUST be `null`.

2.  **Check for Follow-ups (if Rule 1 doesn't apply):**
    - If the new question is a direct follow-up about a device from the history (e.g., it uses words like "its", "these", "that device"), then carry over the `device` name from the history.

3.  **Default:**
    - If none of the above rules match, the `device` filter is `null`.

4.  **Year Filter:**
    - Always extract the `year` if it's mentioned in the new question or the history. If no year is relevant, use `null`.

**Respond with ONLY a valid JSON object.**

Conversation History:
{conversation_history}
---
New Question: "{query}"
---
JSON response:"""
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "format": "json"},
            timeout=OLLAMA_TIMEOUT
        )
        response.raise_for_status()
        response_text = response.json().get("response", "{}")
        return json.loads(response_text)
    except Exception as e:
        print(f"Filter extraction failed: {e}")
        return {}

def generate_and_stream_response(prompt: str):
    """Handles Ollama streaming and returns the final response string."""
    with st.chat_message("assistant", avatar="💡"):
        placeholder = st.empty()
        full_response = ""
        try:
            url = f"{OLLAMA_HOST}/api/generate"
            payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": True}
            with requests.post(url, json=payload, stream=True, timeout=OLLAMA_TIMEOUT) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if line:
                        try:
                            obj = json.loads(line)
                            if not obj.get("done"):
                                full_response += obj.get("response", "")
                                placeholder.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            continue
            placeholder.markdown(full_response)
            st.session_state.chat_history.append(("assistant", full_response))
            return full_response  # <-- NEW
        except requests.exceptions.RequestException as e:
            st.error(f"Ollama connection error: {e}")
            return ""  # <-- NEW

# ========= Main App Logic (Simplified & Reliable) =========

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, content in st.session_state.chat_history:
    with st.chat_message(role, avatar="👤" if role == "user" else "💡"):
        st.markdown(content)

if user_input := st.chat_input("Ask about top devices, common causes, or for examples..."):
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    # --- New Simplified Agent Logic ---
    analysis_type = get_analysis_type_from_query(user_input)
    conversation_str = "\n".join([f"{r}: {c}" for r, c in st.session_state.chat_history[-6:-1]])
    
    if analysis_result := None:
        pass  # just to define the name; no-op so linters don't complain

    if analysis_type:
        st.caption(f"Action: `run_analysis_tool` (type: `{analysis_type}`)")
        with st.spinner("Thinking and analyzing data..."):
            filters = extract_filters_from_conversation(user_input, conversation_str)
            analysis_result = run_analysis_tool(
                analysis_type=analysis_type,
                filter_by_device=filters.get("device"),
                filter_by_year=filters.get("year")
            )
            numeric_answer_log = ""
            if analysis_result.get("analysis_performed") == "top_device_by_failure":
                numeric_answer_log = str(analysis_result.get("count", ""))

        if analysis_result.get("error"):
            st.error(analysis_result["error"])
        else:
            narration_prompt = f"""You are a helpful data analyst. Your analysis tool returned the following JSON data.
Directly narrate the main finding in a clear, human-friendly paragraph.
Then, under a "For example:" heading, list the examples provided in the JSON to illustrate the finding.

Analysis Result:
{json.dumps(analysis_result, indent=2)}

Summary and Examples:"""
            answer_text = generate_and_stream_response(narration_prompt)

            # Log after we have the narration
            log_eval_row({
                "query_id": f"{int(time.time())}",
                "query": user_input,
                "candidate_answer": answer_text,
                "retrieved_ids": "",  # analytics path = no retrieval list
                "cited_ids": "",
                "numeric_answer_log": numeric_answer_log,
                "context_turn": len(st.session_state.chat_history)
            })
    
    else:  # Default to descriptive search if no analysis keywords are found
        # Guardrail: check for unknown devices before searching.
        with st.spinner("Thinking..."):
            filters = extract_filters_from_conversation(user_input, conversation_str)
        
        device_filter = filters.get("device")
        if device_filter and device_filter not in ALLOWED_BRANDS:
            # Use fuzzy matching to see if it's a typo
            best_match, score = process.extractOne(device_filter, ALLOWED_BRANDS)
            if score > 80:
                st.warning(f"Did you mean '{best_match}'? I don't have data for '{device_filter}'.")
                if st.button(f"Use '{best_match}'", key="use_device_suggestion"):
                    device_filter = best_match
                    filters["device"] = best_match
                    st.info(f"Using '{best_match}'. Searching now…")
            else:
                st.error(f"I don't have data for a device named '{device_filter}'.")
        else:
            st.caption("Action: `search_for_examples`")
            with st.spinner("Searching reports for examples..."):
                # If you want retrieval to respect year scoping, pass year=filters.get("year")
                context, retrieved_ids, metas = search_for_examples(query=user_input, return_meta=True)

            if not context:
                st.warning("I couldn't find any relevant information for your query.")
            else:
                final_prompt = f"""The user asked: '{user_input}'. Based on the following context, provide a few clear, bullet-pointed examples that answer the question.

Context:
{context}

Answer:"""
                answer_text = generate_and_stream_response(final_prompt)

                # pick cited ids (top-3 retrieved as a proxy unless you add inline citations)
                cited_ids = retrieved_ids[:3]

                log_eval_row({
                    "query_id": f"{int(time.time())}",
                    "query": user_input,
                    "candidate_answer": answer_text,
                    "retrieved_ids": ";".join(retrieved_ids),
                    "cited_ids": ";".join(cited_ids),
                    "numeric_answer_log": "",
                    "context_turn": len(st.session_state.chat_history)
                })