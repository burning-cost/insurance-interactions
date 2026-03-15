"""Neural Interaction Detection (NID) from trained CANN weights.

Algorithm from Tsang, Cheng and Liu (2018), "Detecting Statistical Interactions
from Neural Network Weights", ICLR 2018, arXiv:1705.04977.

Applied to insurance in Lindström and Palmquist (2023), European Actuarial Journal.

Theory
------
Two features can only interact in a feedforward MLP if they both contribute to the
same first-layer hidden unit. The NID score for a pair (i, j) is:

    d(i, j) = Σ_s  z_s * min(|W1[s,i]|, |W1[s,j]|)

where z_s is the "output importance" of hidden unit s - the cumulative product of
absolute weight matrices from layer 2 to the output, which gives how much unit s
influences the final prediction. The min aggregation is critical: it forces both
features to have non-negligible weight into unit s, ensuring genuine co-participation
rather than one dominant feature inflating the score.

For higher-order interactions (3-way), μ = min across all features in the set.

When working with an ensemble of CANN runs, scores are averaged across runs to
reduce sensitivity to training randomness.
"""

from __future__ import annotations

from itertools import combinations
from typing import NamedTuple

import numpy as np
import polars as pl


class InteractionScore(NamedTuple):
    """A single pairwise (or higher-order) interaction score."""

    features: tuple[str, ...]
    nid_score: float
    nid_score_normalised: float


def _compute_z_scores(w1: np.ndarray, w_rest: list[np.ndarray]) -> np.ndarray:
    """Compute output importance z_s for each first-layer hidden unit.

    z_s = |w_out^T| @ |W_L| @ ... @ |W_2|

    where each multiplication uses absolute values. The result has length = n_hidden_0,
    the number of first-layer units.

    Parameters
    ----------
    w1:
        Weight matrix of first hidden layer, shape (n_hidden_0, n_inputs).
        Not used here - this is W^(1). We need what comes after.
    w_rest:
        Weight matrices from layer 2 to output, each shape (out, in).
        For a 2-layer MLP: [W^(2), W^(out)].
        The last element must be the output layer, shape (1, n_hidden_last).
    """
    if not w_rest:
        # Single-layer network: every hidden unit directly connects to output
        # z_s = 1 for all s (uniform output importance)
        return np.ones(w1.shape[0])

    # Start from the output layer and propagate back using absolute values
    # Output layer: shape (1, n_hidden_last) → squeeze to (n_hidden_last,)
    z = np.abs(w_rest[-1]).sum(axis=0)  # shape (n_hidden_last,)

    # Intermediate layers from second-to-last back to second layer
    for w in reversed(w_rest[:-1]):
        # w shape: (n_out, n_in); z shape: (n_out,)
        # New z: shape (n_in,) = |w|^T @ z
        z = np.abs(w).T @ z

    # z now has length = n_hidden_0 (first hidden layer units)
    return z


def _nid_scores_single(
    w1: np.ndarray,
    w_rest: list[np.ndarray],
    feature_slices: dict[str, slice],
    max_order: int = 2,
) -> dict[tuple[str, ...], float]:
    """NID scores for all feature interaction sets from a single CANN run.

    For categorical features with multiple one-hot columns, the feature-level
    weight is the L2 norm across the one-hot columns for that feature. This
    collapses the variable back to a single importance per feature per hidden unit,
    which is the correct approach since the feature is conceptually one variable.

    Computes interactions of all orders from 2 up to max_order. So max_order=3
    returns both pairwise and three-way interaction scores.
    """
    feature_names = list(feature_slices.keys())
    n_hidden = w1.shape[0]

    z = _compute_z_scores(w1, w_rest)

    # Aggregate W1 columns by feature (L2 norm over one-hot columns)
    # w1_agg shape: (n_hidden, n_features)
    n_features = len(feature_names)
    w1_agg = np.zeros((n_hidden, n_features), dtype=np.float64)
    for k, name in enumerate(feature_names):
        s = feature_slices[name]
        cols = np.abs(w1[:, s])  # shape (n_hidden, n_one_hot_cols)
        w1_agg[:, k] = np.linalg.norm(cols, axis=1)

    # Compute NID score for every candidate interaction set.
    # Includes all orders from 2 to max_order (pairwise + three-way if max_order=3).
    scores: dict[tuple[str, ...], float] = {}
    for order in range(2, max_order + 1):
        for indices in combinations(range(n_features), order):
            feat_tuple = tuple(feature_names[i] for i in indices)
            # Per-unit score: z_s * min(|W1[s,i]|, |W1[s,j]|, ...)
            min_weights = np.min(w1_agg[:, list(indices)], axis=1)  # shape (n_hidden,)
            score = float(np.sum(z * min_weights))
            scores[feat_tuple] = score

    return scores


def compute_nid_scores(
    weight_matrices: list[tuple[np.ndarray, list[np.ndarray]]],
    feature_slices: dict[str, slice],
    max_order: int = 2,
    normalise: bool = True,
) -> list[InteractionScore]:
    """Compute NID interaction scores, averaging over ensemble members.

    Parameters
    ----------
    weight_matrices:
        List of (W1, [W2, ..., W_out]) tuples from ``CANN.get_weight_matrices()``.
    feature_slices:
        Maps original feature name to column slice in the encoded input, from
        ``CANN.feature_slices``.
    max_order:
        Maximum interaction order. 2 = pairwise only (default). 3 = pairwise
        plus three-way interactions.
    normalise:
        If True, normalise scores to [0, 1] by dividing by the maximum. This makes
        cross-run and cross-dataset comparison easier.

    Returns
    -------
    Ranked list of InteractionScore, highest score first.
    """
    all_run_scores: list[dict[tuple[str, ...], float]] = []
    for w1, w_rest in weight_matrices:
        run_scores = _nid_scores_single(w1, w_rest, feature_slices, max_order)
        all_run_scores.append(run_scores)

    # Average across ensemble runs
    all_keys = set().union(*[s.keys() for s in all_run_scores])
    avg_scores: dict[tuple[str, ...], float] = {}
    for key in all_keys:
        vals = [s.get(key, 0.0) for s in all_run_scores]
        avg_scores[key] = float(np.mean(vals))

    # Normalise
    if normalise and avg_scores:
        max_score = max(avg_scores.values())
        if max_score > 0:
            norm_scores = {k: v / max_score for k, v in avg_scores.items()}
        else:
            norm_scores = dict(avg_scores)
    else:
        norm_scores = {k: v for k, v in avg_scores.items()}

    results = [
        InteractionScore(
            features=k,
            nid_score=avg_scores[k],
            nid_score_normalised=norm_scores[k],
        )
        for k in avg_scores
    ]
    results.sort(key=lambda x: x.nid_score, reverse=True)
    return results


def nid_to_dataframe(
    scores: list[InteractionScore],
    order: int | None = None,
) -> pl.DataFrame:
    """Convert NID scores to a Polars DataFrame for easy inspection.

    When the score list contains mixed-order interactions (e.g. both pairwise and
    three-way scores from ``max_order=3``), use the ``order`` parameter to filter
    to a single order before converting. If ``order`` is None and the list contains
    scores of a single order, the column layout is chosen automatically:

    - Order 2 (pairwise): columns are ``feature_1``, ``feature_2``,
      ``nid_score``, ``nid_score_normalised``.
    - Order > 2: columns are ``features`` (list), ``nid_score``,
      ``nid_score_normalised``.

    Parameters
    ----------
    scores:
        Output of ``compute_nid_scores()``.
    order:
        If provided, only include scores where ``len(features) == order``.
        Use ``order=2`` to get just pairwise interactions from a mixed list.
    """
    if not scores:
        return pl.DataFrame()

    # Filter to the requested order if specified
    if order is not None:
        scores = [s for s in scores if len(s.features) == order]
        if not scores:
            return pl.DataFrame()

    # Detect whether the (filtered) list is uniform-order or mixed
    orders_present = {len(s.features) for s in scores}
    if len(orders_present) > 1:
        # Mixed orders — cannot use the flat feature_1/feature_2 layout.
        # Store all interactions in the generic list column.
        return pl.DataFrame(
            {
                "features": [list(s.features) for s in scores],
                "order": [len(s.features) for s in scores],
                "nid_score": [s.nid_score for s in scores],
                "nid_score_normalised": [s.nid_score_normalised for s in scores],
            }
        )

    detected_order = next(iter(orders_present))
    if detected_order == 2:
        return pl.DataFrame(
            {
                "feature_1": [s.features[0] for s in scores],
                "feature_2": [s.features[1] for s in scores],
                "nid_score": [s.nid_score for s in scores],
                "nid_score_normalised": [s.nid_score_normalised for s in scores],
            }
        )
    else:
        return pl.DataFrame(
            {
                "features": [list(s.features) for s in scores],
                "nid_score": [s.nid_score for s in scores],
                "nid_score_normalised": [s.nid_score_normalised for s in scores],
            }
        )
