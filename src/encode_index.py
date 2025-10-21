import os, json, argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from utils import read_corpus_tsv, l2_normalize, ensure_dir, save_mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="TSV: id \\t title \\t text")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    print(f"Loading corpus from {args.corpus} ...")
    df = read_corpus_tsv(args.corpus)
    texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
    ids = df["id"].tolist()

    print(f"Loading model: {args.model}")
    model = SentenceTransformer(args.model)

    print("Encoding documents ...")
    embs = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True, normalize_embeddings=False)
    embs = np.asarray(embs, dtype=np.float32)
    embs = l2_normalize(embs)  # use cosine via inner product

    # Save embeddings for later compression
    npy_path = os.path.join(args.out_dir, "embeddings.npy")
    np.save(npy_path, embs)
    print(f"Saved embeddings: {npy_path}")

    # Build FAISS inner-product index
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)
    faiss_path = os.path.join(args.out_dir, "index.faiss")
    faiss.write_index(index, faiss_path)
    print(f"Saved FAISS index: {faiss_path}")

    # Save mapping for lookup
    mapping = {"ids": ids, "titles": df["title"].tolist(), "texts": df["text"].tolist()}
    map_path = os.path.join(args.out_dir, "mapping.json")
    save_mapping(map_path, mapping)
    print(f"Saved mapping: {map_path}")

if __name__ == "__main__":
    main()
