"""Ranking metrics: Top-K, MRR, MAP@K, F-max and hackathon rank points.

Rank points follow the challenge description: 100 points when the top-ranked
prediction is correct, partial credit when the true variant is inside the top
10, and half credit when only one member of a heterozygous pair is found.
The exact partial-credit curve is not published, so a documented linear
approximation is used (100 - 10 * (rank - 1) inside the top 10).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from rareai.variants.normalizer import normalize_chrom

MAX_SUBMISSION_ROWS = 10


def variant_key(chrom: str | None, pos: int | None, ref: str | None, alt: str | None) -> str | None:
    if chrom is None or pos is None or not ref or not alt:
        return None
    return f"{normalize_chrom(str(chrom))}:{int(pos)}:{str(ref).upper()}>{str(alt).upper()}"


@dataclass
class TruthSet:
    """Confirmed causal variant(s); ``partner_*`` denotes a compound-het pair."""

    primary: str  # variant key
    partner: str | None = None

    @classmethod
    def from_pairs(cls, pairs: list[tuple[str, str | None]]) -> list[TruthSet]:
        return [cls(primary=p, partner=s) for p, s in pairs]


@dataclass
class Metrics:
    topk: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    map_at_10: float = 0.0
    f_max: float = 0.0
    rank_points: float = 0.0
    best_ranks: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "topk": self.topk,
            "mrr": round(self.mrr, 4),
            "map_at_10": round(self.map_at_10, 4),
            "f_max": round(self.f_max, 4),
            "rank_points": round(self.rank_points, 2),
            "best_ranks": self.best_ranks,
        }


def _is_hit(key: str | None, truth: TruthSet) -> bool:
    return key is not None and key in (truth.primary, truth.partner)


def topk_accuracy(ranked_keys: list[str | None], truth: TruthSet, k: int) -> float:
    top = ranked_keys[:k]
    return 1.0 if any(_is_hit(key, truth) for key in top) else 0.0


def reciprocal_rank(ranked_keys: list[str | None], truth: TruthSet) -> float:
    for rank, key in enumerate(ranked_keys, start=1):
        if _is_hit(key, truth):
            return 1.0 / rank
    return 0.0


def average_precision_at_10(ranked_keys: list[str | None], truth: TruthSet) -> float:
    n_targets = 2 if truth.partner else 1
    hits = 0
    total = 0.0
    for rank, key in enumerate(ranked_keys[:10], start=1):
        if _is_hit(key, truth):
            hits += 1
            total += hits / rank
    return total / min(n_targets, 10) if n_targets else 0.0


def f_max(ranked: pl.DataFrame, truth: TruthSet, score_col: str = "score") -> float:
    """Max F1 over score thresholds (challenge secondary metric)."""
    if ranked.is_empty():
        return 0.0
    scores = ranked.sort(score_col, descending=True)[score_col].to_list()
    keys = [
        variant_key(row["chrom"], row["pos"], row["ref"], row["alt"])
        for row in ranked.sort(score_col, descending=True).iter_rows(named=True)
    ]
    n_targets = 2 if truth.partner else 1
    best = 0.0
    hits = 0
    seen_keys: set[str] = set()
    for idx, (_score, key) in enumerate(zip(scores, keys, strict=True), start=1):
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if _is_hit(key, truth):
            hits += 1
        precision = hits / idx
        recall = hits / n_targets
        if precision + recall > 0:
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def rank_points(ranked_keys: list[str | None], truth: TruthSet) -> float:
    """Approximation of the challenge rank-points rule (see module docstring)."""
    full_pairs = {truth.primary, truth.partner} - {None}
    first_hit_rank: int | None = None
    matched: set[str] = set()
    for rank, key in enumerate(ranked_keys[:MAX_SUBMISSION_ROWS], start=1):
        if _is_hit(key, truth) and key is not None:
            if first_hit_rank is None:
                first_hit_rank = rank
            matched.add(key)
    if first_hit_rank is None:
        return 0.0
    points = 100.0 if first_hit_rank == 1 else max(0.0, 100.0 - 10.0 * (first_hit_rank - 1))
    if truth.partner and matched != full_pairs:
        points *= 0.5  # only one member of the heterozygous pair identified
    return points


def evaluate_ranking(ranked: pl.DataFrame, truths: list[TruthSet]) -> Metrics:
    """Evaluate a ranked DataFrame against one or more confirmed variants."""
    ordered = ranked.sort(
        ["score", "rank"] if "rank" in ranked.columns else ["score"], descending=True
    )
    ranked_keys = [
        variant_key(r["chrom"], r["pos"], r["ref"], r["alt"]) for r in ordered.iter_rows(named=True)
    ]

    metrics = Metrics()
    for k in (1, 5, 10):
        metrics.topk[str(k)] = sum(topk_accuracy(ranked_keys, t, k) for t in truths) / max(
            len(truths), 1
        )
    metrics.mrr = sum(reciprocal_rank(ranked_keys, t) for t in truths) / max(len(truths), 1)
    metrics.map_at_10 = sum(average_precision_at_10(ranked_keys, t) for t in truths) / max(
        len(truths), 1
    )
    metrics.f_max = sum(f_max(ordered, t) for t in truths) / max(len(truths), 1)
    metrics.rank_points = sum(rank_points(ranked_keys, t) for t in truths) / max(len(truths), 1)
    for t in truths:
        for rank, key in enumerate(ranked_keys, start=1):
            if _is_hit(key, t):
                metrics.best_ranks.append(f"{t.primary}@rank{rank}")
                break
        else:
            metrics.best_ranks.append(f"{t.primary}@not_found")
    return metrics
