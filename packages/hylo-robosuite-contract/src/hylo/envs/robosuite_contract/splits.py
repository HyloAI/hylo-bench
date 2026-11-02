"""Seeded evaluation splits for robosuite-family benchmarks (pure): a seed fixes an episode (the adapter reseeds the
simulator's generator with it), so splits are disjoint seed ranges, as in the other Hylo benchmarks."""

from __future__ import annotations

RSI_SEED_OFFSET = 30_000          # RFC-0002 splits: their own seeds, disjoint from train / validation / heldout
VALIDATION_SEED_OFFSET = 10_000


def seeded_splits(protocol: str = "rsi", train: int = 10, validation: int = 20, heldout: int = 20,
                  rsi_states: int = 50, gate_looks: int = 1) -> dict[str, tuple[int, ...]]:
    """protocol "rsi": RFC-0002 search / search_val / gate / heldout (40 / 20 / 20 / 20 percent of `rsi_states`
    seeds from 30000) plus a `validation` split (seeds from 10000) for plain evaluation; protocol "standard":
    train (0..) / validation (10000..) / heldout (20000..)."""
    from hylo.core.benchmark import rsi_splits, standard_splits

    if protocol == "standard":
        return standard_splits(train, validation, heldout)
    if protocol != "rsi":
        raise ValueError(f"protocol must be 'rsi' or 'standard', got {protocol!r}")
    n = rsi_states
    raw = rsi_splits(n, (int(0.4 * n), int(0.6 * n), int(0.8 * n)), gate_looks=gate_looks)
    out = {k: tuple(RSI_SEED_OFFSET + s for s in v) for k, v in raw.items()}
    out["validation"] = tuple(range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + validation))
    return out


__all__ = ["RSI_SEED_OFFSET", "VALIDATION_SEED_OFFSET", "seeded_splits"]
