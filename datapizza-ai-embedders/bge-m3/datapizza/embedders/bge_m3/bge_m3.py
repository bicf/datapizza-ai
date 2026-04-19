from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod

from datapizza.core.embedder import BaseEmbedder

log = logging.getLogger(__name__)


# ============================================================
# Abstract base — owns the model, defines the contract
# ============================================================


class BGEM3Embedder(BaseEmbedder):
    """
    Abstract base embedder wrapping BAAI/bge-m3.
    Lazy client init via _set_client / _set_a_client.

    Subclasses implement _extract() to pick the retrieval mode:
      - BGEM3DenseEmbedder   → dense vectors  (1024-dim)
      - BGEM3SparseEmbedder  → lexical weights (dict)
      - BGEM3ColbertEmbedder → multi-vectors   (N×1024)
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-m3",
        device: str = "cuda",
        use_fp16: bool = True,
        batch_size: int = 16,
        max_length: int = 8192,
    ):
        super().__init__(model_name)
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length

        self.client = None
        self.a_client = None

    # ----------------------------------------------------------
    # Lazy init
    # ----------------------------------------------------------

    def _set_client(self) -> None:
        if not self.client:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as e:
                raise ImportError(
                    "FlagEmbedding is required. Install it with: pip install flagembedding"
                ) from e
            log.debug("Loading BGE-M3 model '%s' on device '%s'", self.model_name, self.device)
            try:
                self.client = BGEM3FlagModel(
                    self.model_name,
                    use_fp16=self.use_fp16,
                    device=self.device,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load BGE-M3 model '{self.model_name}' on device '{self.device}': {e}"
                ) from e

    def _set_a_client(self) -> None:
        # BGE-M3 has no native async API — reuse sync model
        if not self.a_client:
            self._set_client()
            self.a_client = self.client

    # ----------------------------------------------------------
    # Internal encode — always returns the full dict
    # ----------------------------------------------------------

    def _encode(
        self,
        texts: list[str],
        return_dense: bool = False,
        return_sparse: bool = False,
        return_colbert: bool = False,
    ) -> dict:
        model = self._get_client()
        return model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

    # ----------------------------------------------------------
    # Subclasses implement these to pick their representation
    # ----------------------------------------------------------

    @abstractmethod
    def _extract(self, output: dict, is_single: bool) -> list[float] | list[list[float]] | dict | list[dict]:
        """Extract the relevant representation from the full encode output."""

    @abstractmethod
    def _encode_for_mode(self, texts: list[str]) -> dict:
        """Call _encode with the right return_* flags for this mode."""

    # ----------------------------------------------------------
    # embed / a_embed
    # ----------------------------------------------------------

    def embed(
        self, text: str | list[str], **kwargs
    ) -> list[float] | list[list[float]] | dict | list[dict]:
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        output = self._encode_for_mode(texts)
        return self._extract(output, is_single)

    async def a_embed(
        self, text: str | list[str], **kwargs
    ) -> list[float] | list[list[float]] | dict | list[dict]:
        self._get_a_client()  # ensure model is loaded before offloading
        return await asyncio.to_thread(self.embed, text)


# ============================================================
# Concrete subclass 1 — Dense (standard vector similarity)
# Output: list[float] (1024-dim) or list[list[float]]
# ============================================================


class BGEM3DenseEmbedder(BGEM3Embedder):
    """
    Dense retrieval mode.
    Returns a 1024-dim float vector per text.
    Use with: Qdrant dense collection, cosine similarity.
    """

    def _encode_for_mode(self, texts: list[str]) -> dict:
        return self._encode(texts, return_dense=True)

    def _extract(self, output: dict, is_single: bool) -> list[float] | list[list[float]]:
        if "dense_vecs" not in output:
            raise KeyError("Expected 'dense_vecs' in encode output for dense mode")
        embeddings = output["dense_vecs"].tolist()
        return embeddings[0] if is_single else embeddings


# ============================================================
# Concrete subclass 2 — Sparse (learned lexical weights)
# Output: dict {token_id: weight} or list[dict]
# ============================================================


class BGEM3SparseEmbedder(BGEM3Embedder):
    """
    Sparse retrieval mode (learned BM25-like).
    Returns a dict of {token_id: weight} per text.
    Use with: Qdrant sparse vectors, hybrid search pipelines.
    """

    def _encode_for_mode(self, texts: list[str]) -> dict:
        return self._encode(texts, return_sparse=True)

    def _extract(self, output: dict, is_single: bool) -> dict | list[dict]:
        if "lexical_weights" not in output:
            raise KeyError("Expected 'lexical_weights' in encode output for sparse mode")
        weights = output["lexical_weights"]
        return weights[0] if is_single else weights


# ============================================================
# Concrete subclass 3 — ColBERT (multi-vector late interaction)
# Output: list[list[float]] (token-level, N×1024) or list of those
# ============================================================


class BGEM3ColbertEmbedder(BGEM3Embedder):
    """
    ColBERT multi-vector mode.
    Returns one 1024-dim vector per token (variable length).
    Use with: Qdrant multi-vector collections, MaxSim scoring.
    """

    def _encode_for_mode(self, texts: list[str]) -> dict:
        return self._encode(texts, return_colbert=True)

    def _extract(
        self, output: dict, is_single: bool
    ) -> list[list[float]] | list[list[list[float]]]:
        if "colbert_vecs" not in output:
            raise KeyError("Expected 'colbert_vecs' in encode output for colbert mode")
        # colbert_vecs is a list of numpy arrays, one per text
        vecs = [v.tolist() for v in output["colbert_vecs"]]
        return vecs[0] if is_single else vecs
