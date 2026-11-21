"""Smoke tests for reference.pt.

Generate it first, from the repo root:
    uv run python reference/dump.py
"""

from pathlib import Path

import pytest
import torch

REF_PATH = Path(__file__).resolve().parent.parent / "reference.pt"

# Qwen3-0.6B config.json values
HIDDEN_SIZE = 1024
VOCAB_SIZE = 151936
NUM_LAYERS = 28
NUM_PROMPT_TOKENS = 5  # "The capital of France is"


@pytest.fixture(scope="module")
def ref():
    if not REF_PATH.exists():
        pytest.fail(f"{REF_PATH} not found. Run: uv run python reference/dump.py")
    return torch.load(REF_PATH)


def test_keys(ref):
    assert set(ref.keys()) == {"ids", "logits", "hidden"}


def test_ids(ref):
    ids = ref["ids"]
    assert ids.shape == (1, NUM_PROMPT_TOKENS)
    assert ids.dtype == torch.int64
    assert (ids >= 0).all() and (ids < VOCAB_SIZE).all()


def test_logits(ref):
    logits = ref["logits"]
    assert logits.shape == (1, NUM_PROMPT_TOKENS, VOCAB_SIZE)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_hidden_states(ref):
    hidden = ref["hidden"]
    # embedding output + one per decoder layer
    assert len(hidden) == NUM_LAYERS + 1
    for i, h in enumerate(hidden):
        assert h.shape == (1, NUM_PROMPT_TOKENS, HIDDEN_SIZE), f"hidden[{i}]"
        assert h.dtype == torch.float32, f"hidden[{i}]"
        assert torch.isfinite(h).all(), f"hidden[{i}]"
