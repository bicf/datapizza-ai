# BGE-m3 Embedder

BAAI/bge-m3 embedder implementation supporting dense, sparse, and ColBERT multi-vector retrieval modes.

## Installation

```bash
pip install datapizza-ai-embedders-bge-m3
```

## Usage

```python
from datapizza.embedders.bge_m3 import BGEM3DenseEmbedder, BGEM3SparseEmbedder, BGEM3ColbertEmbedder

# ---- Dense ----
dense = BGEM3DenseEmbedder(device="cuda", batch_size=16)

single: list[float]       = dense.embed("Il Colosseo è a Roma")          # 1024-dim
batch:  list[list[float]] = dense.embed(["testo uno", "testo due"])

# async
vector = await dense.a_embed("query italiana")


# ---- Sparse ----
sparse = BGEM3SparseEmbedder(device="cuda")

weights: dict            = sparse.embed("Il Colosseo è a Roma")   # {token_id: weight}
batch_w: list[dict]      = sparse.embed(["testo uno", "testo due"])


# ---- ColBERT ----
colbert = BGEM3ColbertEmbedder(device="cuda")

tokens: list[list[float]]       = colbert.embed("Il Colosseo è a Roma")   # (N, 1024)
batch_t: list[list[list[float]]] = colbert.embed(["testo uno", "testo due"])

```
