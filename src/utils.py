import json, os, math, time
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd

def read_corpus_tsv(path: str) -> pd.DataFrame:
    # Expect columns: id \t title \t text
    df = pd.read_csv(path, sep="\t", names=["id", "title", "text"], dtype=str)
    return df

def read_queries_tsv(path: str) -> pd.DataFrame:
    # Expect columns: qid \t query_text
    df = pd.read_csv(path, sep="\t", names=["qid", "query_text"], dtype=str)
    return df

def read_qrels_tsv(path: str) -> Dict[str, List[str]]:
    # Each line: qid \t doc_id
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            qid, did = line.split("\t")
            qrels.setdefault(qid, []).append(did)
    return qrels

def l2_normalize(mat: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(mat, ord=2, axis=axis, keepdims=True)
    return mat / np.maximum(norm, eps)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_mapping(path: str, mapping: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

def load_mapping(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def ndcg_at_k(rels: List[int], k: int = 10) -> float:
    """rels: list of binary relevance for ranked list (1 if relevant, else 0)"""
    def dcg(scores):
        return sum((s / math.log2(i+2)) for i, s in enumerate(scores[:k]))
    ideal = sorted(rels, reverse=True)
    idcg = dcg(ideal)
    if idcg == 0:
        return 0.0
    return dcg(rels) / idcg

def recall_at_k(rels: List[int], total_relevant: int, k: int = 10) -> float:
    hits = sum(rels[:k])
    if total_relevant == 0:
        return 0.0
    return hits / total_relevant

def timing(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        return res, (time.time() - start) * 1000.0
    return wrapper
