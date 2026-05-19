"""Phase 6: PagedKVCache equivalence + capacity tests."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import llmengine
from fixtures import make_tiny_llama_weights


@pytest.fixture(scope="module")
def tiny_engine(tmp_path_factory):
    out = tmp_path_factory.mktemp("paged_kv")
    cfg, _ = make_tiny_llama_weights(out)
    return llmengine.Engine(str(out)), cfg


def test_paged_eq_contiguous_tiny(tiny_engine) -> None:
    """Forward through PagedKVCache must produce bit-identical logits to the
    ContiguousKVCache path — the math is the same, only the K/V storage
    differs."""
    engine, cfg = tiny_engine

    mgr = llmengine.BlockManager(
        num_blocks=8, block_size=4,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    assert mgr.free_blocks == 8

    rng = np.random.default_rng(0)
    for _ in range(10):
        T = int(rng.integers(2, 16))
        ids = rng.integers(0, cfg.vocab_size, size=T).tolist()
        contig = engine.forward_logits(ids)
        paged  = engine.forward_logits_paged(ids, mgr)
        np.testing.assert_array_equal(contig, paged)

    # release_all has happened inside forward_logits_paged — pool drains.
    assert mgr.free_blocks == 8


def test_paged_alloc_at_block_boundaries(tiny_engine) -> None:
    """block_size=4 means a 9-token forward should consume exactly 3 blocks
    (positions 0-3, 4-7, 8). After the call, the pool drains back."""
    engine, cfg = tiny_engine
    mgr = llmengine.BlockManager(
        num_blocks=8, block_size=4,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )

    ids = list(range(9))
    engine.forward_logits_paged(ids, mgr)
    assert mgr.free_blocks == 8


def test_paged_capacity_exhaustion(tiny_engine) -> None:
    """Tiny pool (1 block × block_size=4 = 4 tokens of capacity). A 6-token
    forward must raise on exhaustion at position 4."""
    engine, cfg = tiny_engine
    mgr = llmengine.BlockManager(
        num_blocks=1, block_size=4,
        num_layers=cfg.num_hidden_layers,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
    )
    with pytest.raises(RuntimeError, match="KV pool exhausted"):
        engine.forward_logits_paged([1, 2, 3, 4, 5, 6], mgr)
    # On exhaustion, forward_logits_paged calls release_all internally.
    assert mgr.free_blocks == 1


# ----- Real-1B paged-vs-contiguous equivalence -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = REPO_ROOT / "data" / "llama-3.2-1b-instruct"


@pytest.mark.skipif(
    not (REAL_MODEL_DIR / "model.safetensors").exists(),
    reason="Real Llama 3.2 1B-Instruct not downloaded",
)
def test_real_1b_paged_eq_contiguous() -> None:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(REAL_MODEL_DIR))
    ids = tok.encode("The capital of France is", add_special_tokens=True)

    engine = llmengine.Engine(str(REAL_MODEL_DIR))
    contig = engine.forward_logits(ids)

    mgr = llmengine.BlockManager(
        num_blocks=4, block_size=4,
        num_layers=engine.cfg.num_hidden_layers,
        num_kv_heads=engine.cfg.num_key_value_heads,
        head_dim=engine.cfg.head_dim,
    )
    paged = engine.forward_logits_paged(ids, mgr)
    np.testing.assert_array_equal(contig, paged)
    assert mgr.free_blocks == 4
