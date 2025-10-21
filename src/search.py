import os, argparse, json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from utils import load_mapping, l2_normalize

def load_index(index_path: str):
    return faiss.read_index(index_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    index = load_index(args.index)
    mapping = load_mapping(args.mapping)

    # Try to load PCA (if compress.py created it in the same folder as the index)
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
        # Safe small default for tiny datasets
        if hasattr(index, "nprobe"):
            index.nprobe = min(8, getattr(index, "nlist", 8))
    except Exception:
        pass

    model = SentenceTransformer(args.model)
    qvec = model.encode([args.query], normalize_embeddings=False)
    qvec = np.asarray(qvec, dtype=np.float32)

    if pca is not None:
        qvec = pca.transform(qvec).astype(np.float32)

    qvec = l2_normalize(qvec)
    D, I = index.search(qvec, args.k)

    print(f"\nQuery: {args.query}\n")
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), start=1):
        doc_id = mapping["ids"][int(idx)]
        title = mapping["titles"][int(idx)]
        text = mapping["texts"][int(idx)]
        snippet = (text[:180] + "...") if len(text) > 180 else text
        print(f"#{rank}  score={score:.4f}  id={doc_id}\n   {title}\n   {snippet}\n")

if __name__ == "__main__":
    main()
