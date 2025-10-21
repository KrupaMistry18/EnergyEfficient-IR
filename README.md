EfficientSearch — Energy-Efficient / Memory-Light Semantic IR

Goal: Build a semantic retrieval pipeline that approaches baseline quality while using less memory and lower latency via PCA and FAISS Product Quantization (PQ).

What’s inside

Baseline dense retrieval (Sentence-Transformers + FAISS flat IP).

Compression options: PCA (dim reduction) and IVFPQ (quantization).

Scripts for encode → compress → search → evaluate.

Toy dataset (5 docs) + a synthetic big dataset (300 docs) so PCA/PQ train and work.

1) Setup

Python (recommended): 3.10 or 3.11

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

2) Repo Structure
efficientsearch/
├─ data/
│  ├─ sample_corpus.tsv
│  ├─ sample_queries.tsv
│  ├─ sample_qrels.tsv
│  ├─ big_corpus.tsv
│  ├─ big_queries.tsv
│  └─ big_qrels.tsv
├─ src/
│  ├─ encode_index.py     # build baseline dense index (SBERT + FAISS)
│  ├─ compress.py         # PCA + (optional) IVFPQ; saves pca.joblib
│  ├─ search.py           # ad-hoc query against any index
│  ├─ evaluate.py         # nDCG@10, Recall@10 vs qrels
│  └─ utils.py            # IO + metrics helpers
├─ requirements.txt
└─ README.md

3) Quickstart (Big Dataset)
A) Build baseline dense index
python src/encode_index.py \
  --corpus data/big_corpus.tsv \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --out_dir artifacts/baseline_big

B) Compress with PCA 128 + PQ m=16
python src/compress.py \
  --embeddings artifacts/baseline_big/embeddings.npy \
  --mapping artifacts/baseline_big/mapping.json \
  --out_dir artifacts/pca128_pq16_big \
  --pca_dims 128 \
  --pq_m 16

C) Search sanity check
python src/search.py \
  --index artifacts/pca128_pq16_big/index.faiss \
  --mapping artifacts/pca128_pq16_big/mapping.json \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --query "neural networks for image recognition" \
  --k 5

D) Evaluate (nDCG@10 & Recall@10)
python src/evaluate.py \
  --index artifacts/pca128_pq16_big/index.faiss \
  --mapping artifacts/pca128_pq16_big/mapping.json \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --queries data/big_queries.tsv \
  --qrels data/big_qrels.tsv \
  --k 10

E) Baseline (for comparison)
python src/evaluate.py \
  --index artifacts/baseline_big/index.faiss \
  --mapping artifacts/baseline_big/mapping.json \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --queries data/big_queries.tsv \
  --qrels data/big_qrels.tsv \
  --k 10


If you used PCA, the scripts auto-load pca.joblib to project query vectors at search/eval time.

4) Experiments to Try (Report-Ready)

PCA-only: 256 / 192 / 128 / 96 / 64 dims (flat index).

PQ-only: m = 16 / 32 on 384-dim (no PCA).

PCA+PQ: e.g., 128-dim + m=16 or m=32.

IVF recall tuning: increase nprobe (e.g., 8 → 16 → 32) to improve Recall.

Record for each config:

Quality: nDCG@10, Recall@10

Efficiency: index size (ls -lh .../index.faiss) and rough latency (shell time on search.py)

Example table columns:

Config | Dim | PQ m | nprobe | nDCG@10 | Recall@10 | Index Size | ~Latency

5) Tips & Gotchas

PCA constraint: pca_dims ≤ min(n_docs, original_dim); with 300 docs, 128 is safe.

PQ constraint: choose m that divides the (post-PCA) dimension.

Small datasets: FAISS may warn about few training points; it’s okay for this demo.

zsh comments: zsh doesn’t ignore # by default; use setopt interactivecomments or omit comment lines when pasting.

6) Why This Improves the “Current System”

Baseline dense retrieval stores full-precision 384-D embeddings and uses a flat index—accurate but heavy.
This repo shows you can compress (PCA, PQ) to cut memory and latency while keeping quality close to baseline, then quantify the trade-offs.