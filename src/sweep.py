# src/sweep.py
import os, json, time, argparse, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from utils import load_mapping, read_queries_tsv, read_qrels_tsv, l2_normalize, ndcg_at_k, recall_at_k
import joblib

def eval_index(index_path, mapping_path, model_name, queries_path, qrels_path, k=10, nprobe=16):
    index = faiss.read_index(index_path)
    if hasattr(index, "nprobe"): index.nprobe = nprobe
    mapping = load_mapping(mapping_path)
    ids = mapping["ids"]
    pca = None
    pca_path = os.path.join(os.path.dirname(index_path), "pca.joblib")
    if os.path.exists(pca_path): pca = joblib.load(pca_path)

    model = SentenceTransformer(model_name)
    Q = read_queries_tsv(queries_path); qrels = read_qrels_tsv(qrels_path)

    ndcgs, recalls = [], []
    for _, row in Q.iterrows():
        qid, qtext = row["qid"], row["query_text"]
        q = model.encode([qtext], normalize_embeddings=False).astype(np.float32)
        if pca is not None: q = pca.transform(q).astype(np.float32)
        q = l2_normalize(q)
        _, I = index.search(q, k)
        ranked_ids = [ids[int(r)] for r in I[0]]
        rel = set(qrels.get(qid, []))
        bin_rel = [1 if rid in rel else 0 for rid in ranked_ids]
        ndcgs.append(ndcg_at_k(bin_rel, k=k))
        recalls.append(recall_at_k(bin_rel, total_relevant=len(rel), k=k))

    return float(np.mean(ndcgs)), float(np.mean(recalls))

def size_bytes(path): return os.path.getsize(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--queries", default="data/big_queries.tsv")
    ap.add_argument("--qrels", default="data/big_qrels.tsv")
    ap.add_argument("--out_csv", default="artifacts/results.csv")
    args = ap.parse_args()

    runs = [
        ("baseline_flat", "artifacts/baseline_big/index.faiss", "artifacts/baseline_big/mapping.json"),
        ("pca128_flat",   "artifacts/pca128_flat_big/index.faiss", "artifacts/pca128_flat_big/mapping.json"),
        ("pq32_only",     "artifacts/pq32_only_big/index.faiss", "artifacts/pq32_only_big/mapping.json"),
        ("pca128_pq16",   "artifacts/pca128_pq16_big/index.faiss", "artifacts/pca128_pq16_big/mapping.json"),
    ]

    os.makedirs("artifacts", exist_ok=True)
    with open(args.out_csv, "w") as f:
        f.write("config,nprobe,ndcg@10,recall@10,index_size_bytes\n")
        for name, idx, mapping in runs:
            if not os.path.exists(idx): 
                print(f"Skip {name} (missing {idx})"); continue
            ndcg, rec = eval_index(idx, mapping, args.model, args.queries, args.qrels, k=10, nprobe=16)
            f.write(f"{name},16,{ndcg:.4f},{rec:.4f},{size_bytes(idx)}\n")
            print(name, ndcg, rec, size_bytes(idx))

if __name__ == "__main__":
    main()
