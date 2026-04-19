from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from datapizza.embedders.bge_m3 import (
    BGEM3ColbertEmbedder,
    BGEM3DenseEmbedder,
    BGEM3SparseEmbedder,
)


# ------------------------------------------------------------------ fixtures


def _make_mock_model(dense=None, sparse=None, colbert=None):
    """Return a MagicMock that behaves like BGEM3FlagModel.encode()."""
    model = MagicMock()
    output = {}
    if dense is not None:
        output["dense_vecs"] = np.array(dense)
    if sparse is not None:
        output["lexical_weights"] = sparse
    if colbert is not None:
        output["colbert_vecs"] = [np.array(v) for v in colbert]
    model.encode.return_value = output
    return model


# ================================================================== init


class TestInit:
    def test_dense_defaults(self):
        e = BGEM3DenseEmbedder()
        assert e.model_name == "BAAI/bge-m3"
        assert e.device == "cuda"
        assert e.use_fp16 is True
        assert e.batch_size == 16
        assert e.max_length == 8192
        assert e.client is None
        assert e.a_client is None

    def test_sparse_custom_params(self):
        e = BGEM3SparseEmbedder(device="cpu", use_fp16=False, batch_size=8, max_length=512)
        assert e.device == "cpu"
        assert e.use_fp16 is False
        assert e.batch_size == 8
        assert e.max_length == 512

    def test_colbert_custom_model_name(self):
        e = BGEM3ColbertEmbedder(model_name="BAAI/bge-m3-custom")
        assert e.model_name == "BAAI/bge-m3-custom"


# ================================================================== lazy init


class TestLazyInit:
    def test_set_client_lazy(self):
        e = BGEM3DenseEmbedder()
        mock_model = MagicMock()
        with patch("datapizza.embedders.bge_m3.bge_m3.BGEM3FlagModel", return_value=mock_model):
            # FlagEmbedding should NOT be imported yet
            assert e.client is None
            e._set_client()
            assert e.client is mock_model

    def test_set_client_idempotent(self):
        e = BGEM3DenseEmbedder()
        mock_model = MagicMock()
        with patch("datapizza.embedders.bge_m3.bge_m3.BGEM3FlagModel", return_value=mock_model):
            e._set_client()
            e._set_client()  # second call should not re-instantiate
        assert mock_model.call_count == 0  # constructor called once via return_value

    def test_set_a_client_reuses_sync_model(self):
        e = BGEM3DenseEmbedder()
        mock_model = MagicMock()
        with patch("datapizza.embedders.bge_m3.bge_m3.BGEM3FlagModel", return_value=mock_model):
            e._set_a_client()
        assert e.a_client is e.client

    def test_missing_flagembedding_raises_import_error(self):
        e = BGEM3DenseEmbedder()
        with patch.dict("sys.modules", {"FlagEmbedding": None}):
            with pytest.raises(ImportError, match="FlagEmbedding is required"):
                e._set_client()

    def test_model_load_failure_raises_runtime_error(self):
        e = BGEM3DenseEmbedder()
        with patch("datapizza.embedders.bge_m3.bge_m3.BGEM3FlagModel", side_effect=RuntimeError("GPU OOM")):
            with pytest.raises(RuntimeError, match="Failed to load BGE-M3 model"):
                e._set_client()


# ================================================================== dense


class TestDenseEmbedder:
    _vec_a = [0.1] * 1024
    _vec_b = [0.2] * 1024

    def _embedder_with_mock(self):
        e = BGEM3DenseEmbedder()
        e.client = _make_mock_model(dense=[self._vec_a, self._vec_b])
        e.a_client = e.client
        return e

    def test_embed_single_returns_list_of_float(self):
        e = self._embedder_with_mock()
        e.client = _make_mock_model(dense=[self._vec_a])
        e.a_client = e.client
        result = e.embed("hello")
        assert isinstance(result, list)
        assert len(result) == 1024
        assert all(isinstance(v, float) for v in result)

    def test_embed_batch_returns_list_of_lists(self):
        e = self._embedder_with_mock()
        result = e.embed(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 1024

    def test_extract_missing_key_raises(self):
        e = BGEM3DenseEmbedder()
        with pytest.raises(KeyError, match="dense_vecs"):
            e._extract({}, is_single=True)

    @pytest.mark.asyncio
    async def test_a_embed_single(self):
        e = BGEM3DenseEmbedder()
        e.client = _make_mock_model(dense=[self._vec_a])
        e.a_client = e.client
        result = await e.a_embed("hello")
        assert len(result) == 1024


# ================================================================== sparse


class TestSparseEmbedder:
    _weights_a = {"token1": 0.9, "token2": 0.3}
    _weights_b = {"token3": 0.7}

    def _embedder_with_mock(self, single=False):
        e = BGEM3SparseEmbedder()
        data = [self._weights_a] if single else [self._weights_a, self._weights_b]
        e.client = _make_mock_model(sparse=data)
        e.a_client = e.client
        return e

    def test_embed_single_returns_dict(self):
        e = self._embedder_with_mock(single=True)
        result = e.embed("hello")
        assert isinstance(result, dict)
        assert result == self._weights_a

    def test_embed_batch_returns_list_of_dicts(self):
        e = self._embedder_with_mock()
        result = e.embed(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0] == self._weights_a

    def test_extract_missing_key_raises(self):
        e = BGEM3SparseEmbedder()
        with pytest.raises(KeyError, match="lexical_weights"):
            e._extract({}, is_single=True)

    @pytest.mark.asyncio
    async def test_a_embed_batch(self):
        e = self._embedder_with_mock()
        result = await e.a_embed(["hello", "world"])
        assert len(result) == 2


# ================================================================== colbert


class TestColbertEmbedder:
    # ColBERT: each text → variable-length sequence of 1024-dim vectors
    _vecs_a = [[0.1] * 1024, [0.2] * 1024]   # 2 tokens
    _vecs_b = [[0.3] * 1024]                   # 1 token

    def _embedder_with_mock(self, single=False):
        e = BGEM3ColbertEmbedder()
        data = [self._vecs_a] if single else [self._vecs_a, self._vecs_b]
        e.client = _make_mock_model(colbert=data)
        e.a_client = e.client
        return e

    def test_embed_single_returns_list_of_vectors(self):
        e = self._embedder_with_mock(single=True)
        result = e.embed("hello")
        assert isinstance(result, list)
        assert len(result) == 2       # 2 tokens
        assert len(result[0]) == 1024

    def test_embed_batch_returns_list_of_list_of_vectors(self):
        e = self._embedder_with_mock()
        result = e.embed(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2
        assert len(result[0]) == 2    # 2 tokens for first text
        assert len(result[1]) == 1    # 1 token for second text

    def test_extract_missing_key_raises(self):
        e = BGEM3ColbertEmbedder()
        with pytest.raises(KeyError, match="colbert_vecs"):
            e._extract({}, is_single=True)

    @pytest.mark.asyncio
    async def test_a_embed_single(self):
        e = self._embedder_with_mock(single=True)
        result = await e.a_embed("hello")
        assert len(result) == 2
