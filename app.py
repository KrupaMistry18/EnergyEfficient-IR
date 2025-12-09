import os
import sys
import time

import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

# Make src/ importable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from utils import load_mapping, l2_normalize  # type: ignore

try:
    import joblib
except ImportError:
    joblib = None

# ---- Configs: name -> paths ----
CONFIGS = {
    "baseline_flat": {
        "label": "Baseline (384-d flat)",
        "index": "artifacts/baseline_big/index.faiss",
        "mapping": "artifacts/baseline_big/mapping.json",
        "has_pca": False,
    },
    "pca128_flat": {
        "label": "PCA 128-d (flat)",
        "index": "artifacts/pca128_flat_big/index.faiss",
        "mapping": "artifacts/pca128_flat_big/mapping.json",
        "has_pca": True,
    },
    "pca128_pq16": {
        "label": "PCA 128-d + PQ m=16",
        "index": "artifacts/pca128_pq16_big/index.faiss",
        "mapping": "artifacts/pca128_pq16_big/mapping.json",
        "has_pca": True,
    },
    "pq32_only": {
        "label": "PQ-only m=32 (384-d)",
        "index": "artifacts/pq32_only_big/index.faiss",
        "mapping": "artifacts/pq32_only_big/mapping.json",
        "has_pca": False,
    },
}


@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return SentenceTransformer(name)


@st.cache_resource(show_spinner=False)
def load_index_and_mapping(index_path: str, mapping_path: str, has_pca: bool):
    index = faiss.read_index(os.path.join(BASE_DIR, index_path))
    mapping = load_mapping(os.path.join(BASE_DIR, mapping_path))

    # Try to load PCA if needed
    pca = None
    if has_pca and joblib is not None:
        pca_path = os.path.join(os.path.dirname(os.path.join(BASE_DIR, index_path)), "pca.joblib")
        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)

    # For IVF / PQ indexes, set nprobe for higher recall
    try:
        if hasattr(index, "nprobe"):
            index.nprobe = min(32, getattr(index, "nlist", 32))
    except Exception:
        pass

    return index, mapping, pca


def run_search(query: str, model_name: str, cfg_key: str, k: int = 10):
    cfg = CONFIGS[cfg_key]
    index, mapping, pca = load_index_and_mapping(
        cfg["index"], cfg["mapping"], cfg["has_pca"]
    )
    model = load_model(model_name)

    qvec = model.encode([query], normalize_embeddings=False)
    qvec = np.asarray(qvec, dtype=np.float32)
    if pca is not None:
        qvec = pca.transform(qvec).astype(np.float32)
    qvec = l2_normalize(qvec)

    start = time.time()
    D, I = index.search(qvec, k)
    elapsed_ms = (time.time() - start) * 1000.0

    ids = mapping["ids"]
    titles = mapping["titles"]
    texts = mapping["texts"]

    results = []
    for score, idx in zip(D[0], I[0]):
        idx = int(idx)
        doc_id = ids[idx]
        title = titles[idx]
        text = texts[idx]
        snippet = text if len(text) <= 220 else text[:220] + "..."
        results.append((doc_id, title, snippet, float(score)))
    return results, elapsed_ms


# ------------------- Streamlit UI ------------------- #

st.title("EfficientSearch Demo")
st.write(
    "Compare baseline dense retrieval vs compressed indexes "
    "(PCA and PQ) on the same corpus."
)

config_key = st.sidebar.selectbox(
    "Index configuration",
    options=list(CONFIGS.keys()),
    format_func=lambda k: CONFIGS[k]["label"],
)

model_name = "sentence-transformers/all-MiniLM-L6-v2"

st.sidebar.markdown("**Model:** `all-MiniLM-L6-v2`")
st.sidebar.markdown(f"**Config key:** `{config_key}`")

query = st.text_input(
    "Enter your query",
    value="neural networks for image recognition",
)

k = st.slider("Top-k results", min_value=3, max_value=20, value=10, step=1)

if st.button("Search") and query.strip():
    with st.spinner("Searching..."):
        results, elapsed_ms = run_search(query.strip(), model_name, config_key, k=k)

    st.markdown(f"**Config:** {CONFIGS[config_key]['label']}")
    st.markdown(f"**Query time:** {elapsed_ms:.2f} ms")

    for rank, (doc_id, title, snippet, score) in enumerate(results, start=1):
        st.markdown(f"### #{rank}: {title}")
        st.caption(f"ID: {doc_id} · Score: {score:.4f}")
        st.write(snippet)
        st.markdown("---")
