import os, argparse
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from utils import read_queries_tsv, read_qrels_tsv, load_mapping, l2_normalize, ndcg_at_k, recall_at_k

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    index = faiss.read_index(args.index)
    mapping = load_mapping(args.mapping)
    id_list = mapping["ids"]

    # Try to load PCA if present (created by compress.py)
    pca = None
    try:
        import joblib
        pca_path = os.path.join(os.path.dirname(args.index), "pca.joblib")
        if os.path.exists(pca_path):
            pca = joblib.load(pca_path)
    except Exception:
        pca = None

    # Optional: set nprobe for IVF/PQ indexes
    try:
        if hasattr(index, "nprobe"):
            index.nprobe = min(8, getattr(index, "nlist", 8))
    except Exception:
        pass

    queries_df = read_queries_tsv(args.queries)
    qrels = read_qrels_tsv(args.qrels)

    model = SentenceTransformer(args.model)

    ndcgs, recalls = [], []
    for _, row in queries_df.iterrows():
        qid = row["qid"]
        qtext = row["query_text"]

        qvec = model.encode([qtext], normalize_embeddings=False)
        qvec = np.asarray(qvec, dtype=np.float32)

        if pca is not None:
            qvec = pca.transform(qvec).astype(np.float32)

        qvec = l2_normalize(qvec)

        D, I = index.search(qvec, args.k)
        ranked_ids = [id_list[int(r)] for r in I[0].tolist()]
        relevant = set(qrels.get(qid, []))

        rels_binary = [1 if rid in relevant else 0 for rid in ranked_ids]
        ndcgs.append(ndcg_at_k(rels_binary, k=args.k))
        recalls.append(recall_at_k(rels_binary, total_relevant=len(relevant), k=args.k))

    mean_ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    print(f"nDCG@{args.k} = {mean_ndcg:.4f}")
    print(f"Recall@{args.k} = {mean_recall:.4f}")

if __name__ == "__main__":
    main()
