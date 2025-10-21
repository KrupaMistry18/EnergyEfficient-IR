import os, argparse, json
import numpy as np
import faiss
from sklearn.decomposition import PCA
from utils import ensure_dir, load_mapping, l2_normalize

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="Path to baseline embeddings.npy (float32)")
    ap.add_argument("--mapping", required=True, help="Path to mapping.json from baseline")
    ap.add_argument("--out_dir", required=True, help="Where to store compressed index")
    ap.add_argument("--pca_dims", type=int, default=0, help="If >0, apply PCA to this dimensionality before indexing")
    ap.add_argument("--pq_m", type=int, default=0, help="If >0, use FAISS Product Quantization with m sub-vectors")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    X = np.load(args.embeddings).astype(np.float32)
    print(f"Loaded embeddings: {X.shape}")

    # Optional PCA
    if args.pca_dims and args.pca_dims > 0 and args.pca_dims < X.shape[1]:
        print(f"Applying PCA -> {args.pca_dims} dims ...")
        pca = PCA(n_components=args.pca_dims, random_state=42)
        X = pca.fit_transform(X).astype(np.float32)
        # Save PCA for query-time transform
        import joblib
        joblib.dump(pca, os.path.join(args.out_dir, "pca.joblib"))
    else:
        print("Skipping PCA.")
        pca = None

    # Re-normalize after PCA
    X = l2_normalize(X)

    d = X.shape[1]
    if args.pq_m and args.pq_m > 0:
        # Use IndexIVFPQ for compression + speed; train on all vectors (toy) or sample
        nlist = min(256, max(16, X.shape[0] // 50))
        m = args.pq_m
        nbits = 8  # 8 bits per sub-quantizer
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)
        print(f"Training IVFPQ with nlist={nlist}, m={m}, nbits={nbits} ...")
        index.train(X)
        index.add(X)
    else:
        print("Using flat IP index (no PQ).")
        index = faiss.IndexFlatIP(d)
        index.add(X)

    faiss_path = os.path.join(args.out_dir, "index.faiss")
    faiss.write_index(index, faiss_path)
    print(f"Saved compressed FAISS index: {faiss_path}")

    # Copy mapping
    with open(args.mapping, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    with open(os.path.join(args.out_dir, "mapping.json"), "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print("Saved mapping.json")

if __name__ == "__main__":
    main()
