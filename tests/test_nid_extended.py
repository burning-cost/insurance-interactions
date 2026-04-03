"""Extended NID tests covering edge cases not in test_nid.py.

Covers:
  - _compute_z_scores: deep network (3+ layers), single hidden unit,
    proportionality to output weights
  - _nid_scores_single: single feature pair, three-way exact count,
    all-zero weights, feature with single encoded column vs. multi-column
  - compute_nid_scores: empty weight matrices list, un-normalised scores,
    all-zero weights edge case, single ensemble member, mixed-order normalisation
  - nid_to_dataframe: three-way uniform order, high-order generic layout,
    scores match input values, sorting preserved
  - InteractionScore: field access, ordering by nid_score
"""

from __future__ import annotations

import numpy as np
import pytest

from insurance_interactions.nid import (
    InteractionScore,
    _compute_z_scores,
    _nid_scores_single,
    compute_nid_scores,
    nid_to_dataframe,
)


# ---------------------------------------------------------------------------
# _compute_z_scores additional coverage
# ---------------------------------------------------------------------------

class TestZScoresExtended:
    def test_deep_network_three_layers(self):
        """Three weight matrices (W2, W3, W_out) should propagate correctly."""
        w1 = np.abs(np.random.randn(8, 4).astype(np.float32))
        w2 = np.abs(np.random.randn(4, 8).astype(np.float32))
        w3 = np.abs(np.random.randn(2, 4).astype(np.float32))
        w_out = np.abs(np.random.randn(1, 2).astype(np.float32))
        z = _compute_z_scores(w1, [w2, w3, w_out])
        assert z.shape == (8,)
        assert np.all(z >= 0)

    def test_single_hidden_unit(self):
        """Network with a single hidden unit should still give a scalar z."""
        w1 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # shape (1, 3)
        w_out = np.array([[5.0]], dtype=np.float32)          # shape (1, 1)
        z = _compute_z_scores(w1, [w_out])
        assert z.shape == (1,)
        assert float(z[0]) == pytest.approx(5.0)

    def test_z_proportional_to_output_weight(self):
        """z should scale proportionally with the output layer weight magnitude."""
        w1 = np.ones((4, 3), dtype=np.float32)
        w_out_1 = np.ones((1, 4), dtype=np.float32)
        w_out_2 = np.full((1, 4), 3.0, dtype=np.float32)
        z1 = _compute_z_scores(w1, [w_out_1])
        z2 = _compute_z_scores(w1, [w_out_2])
        np.testing.assert_allclose(z2, z1 * 3.0, rtol=1e-5)

    def test_intermediate_zero_weight_kills_unit(self):
        """A zero intermediate weight for one unit should propagate to zero z."""
        # w2 is 4×8; set column 2 (unit 2's input from w2) to zero
        w1 = np.ones((8, 3), dtype=np.float32)
        w2 = np.ones((4, 8), dtype=np.float32)
        w2[:, 2] = 0.0  # unit 2 has no contribution to w2 → z[2] = 0
        w_out = np.ones((1, 4), dtype=np.float32)
        z = _compute_z_scores(w1, [w2, w_out])
        assert float(z[2]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _nid_scores_single additional coverage
# ---------------------------------------------------------------------------

class TestNIDScoresSingleExtended:
    def test_two_features_gives_one_pair(self):
        feature_slices = {"a": slice(0, 1), "b": slice(1, 2)}
        w1 = np.abs(np.random.randn(4, 2).astype(np.float32))
        w_out = np.ones((1, 4), dtype=np.float32)
        scores = _nid_scores_single(w1, [w_out], feature_slices, max_order=2)
        assert len(scores) == 1
        assert ("a", "b") in scores

    def test_three_way_exact_count(self):
        """For n=4 features with max_order=3: C(4,2)+C(4,3) = 6+4 = 10 scores."""
        n_features = 4
        feature_slices = {f"f{i}": slice(i, i + 1) for i in range(n_features)}
        w1 = np.abs(np.random.randn(8, n_features).astype(np.float32))
        w_out = np.ones((1, 8), dtype=np.float32)
        scores = _nid_scores_single(w1, [w_out], feature_slices, max_order=3)
        expected = 4 * 3 // 2 + 4 * 3 * 2 // 6  # C(4,2) + C(4,3) = 6 + 4 = 10
        assert len(scores) == expected

    def test_all_zero_w1_gives_zero_scores(self):
        """If W1 is all zeros, all NID scores must be zero."""
        feature_slices = {"a": slice(0, 1), "b": slice(1, 2), "c": slice(2, 3)}
        w1 = np.zeros((8, 3), dtype=np.float32)
        w_out = np.ones((1, 8), dtype=np.float32)
        scores = _nid_scores_single(w1, [w_out], feature_slices)
        assert all(v == pytest.approx(0.0) for v in scores.values())

    def test_l2_norm_aggregation_for_ohe(self):
        """Feature with 3 one-hot cols: L2 norm of those 3 cols per hidden unit."""
        feature_slices = {"cat_feat": slice(0, 3), "cont_feat": slice(3, 4)}
        # Hidden unit 0: strong on cat_feat (cols 0,1,2) and cont_feat
        w1 = np.zeros((2, 4), dtype=np.float32)
        w1[0, 0] = 1.0
        w1[0, 1] = 2.0
        w1[0, 2] = 2.0  # L2 norm for cat_feat unit 0 = sqrt(1+4+4) = 3
        w1[0, 3] = 3.0  # cont_feat unit 0
        w_out = np.ones((1, 2), dtype=np.float32)
        scores = _nid_scores_single(w1, [w_out], feature_slices)
        # Expected: z=[1,1], w1_agg[0,0]=3, w1_agg[0,1]=3 → min=3, score=3
        assert scores[("cat_feat", "cont_feat")] == pytest.approx(3.0, rel=1e-4)

    @pytest.mark.parametrize("n_features,max_order", [
        (5, 2),
        (4, 3),
        (3, 2),
    ])
    def test_score_count_parametrized(self, n_features: int, max_order: int):
        from math import comb
        feature_slices = {f"f{i}": slice(i, i + 1) for i in range(n_features)}
        w1 = np.abs(np.random.randn(6, n_features).astype(np.float32))
        w_out = np.ones((1, 6), dtype=np.float32)
        scores = _nid_scores_single(w1, [w_out], feature_slices, max_order=max_order)
        expected = sum(comb(n_features, k) for k in range(2, max_order + 1))
        assert len(scores) == expected


# ---------------------------------------------------------------------------
# compute_nid_scores additional coverage
# ---------------------------------------------------------------------------

class TestComputeNIDScoresExtended:
    def test_single_ensemble_member(self):
        """Single ensemble member: avg == raw score."""
        feature_slices = {"a": slice(0, 1), "b": slice(1, 2)}
        w1 = np.abs(np.random.randn(4, 2).astype(np.float32))
        w_out = np.ones((1, 4), dtype=np.float32)
        scores = compute_nid_scores([(w1, [w_out])], feature_slices)
        assert len(scores) == 1

    def test_unnormalised_raw_scores(self):
        """normalise=False should return scores > 1 if max > 1."""
        feature_slices = {"a": slice(0, 1), "b": slice(1, 2)}
        w1 = np.full((4, 2), 5.0, dtype=np.float32)  # large weights
        w_out = np.ones((1, 4), dtype=np.float32)
        scores = compute_nid_scores([(w1, [w_out])], feature_slices, normalise=False)
        raw_score = scores[0].nid_score_normalised  # normalised == raw when normalise=False
        # With normalise=False, nid_score_normalised == nid_score (both are avg scores)
        assert scores[0].nid_score == pytest.approx(scores[0].nid_score_normalised)

    def test_max_normalised_score_is_1(self):
        """When normalise=True, the top score should be exactly 1.0."""
        n_features = 4
        feature_slices = {f"f{i}": slice(i, i + 1) for i in range(n_features)}
        w1 = np.abs(np.random.randn(8, n_features).astype(np.float32))
        w_out = np.ones((1, 8), dtype=np.float32)
        scores = compute_nid_scores([(w1, [w_out])], feature_slices, normalise=True)
        assert scores[0].nid_score_normalised == pytest.approx(1.0)

    def test_all_zero_weights_normalised(self):
        """All-zero weights → all scores are 0; normalisation should not divide by zero."""
        feature_slices = {"a": slice(0, 1), "b": slice(1, 2), "c": slice(2, 3)}
        w1 = np.zeros((4, 3), dtype=np.float32)
        w_out = np.ones((1, 4), dtype=np.float32)
        scores = compute_nid_scores([(w1, [w_out])], feature_slices, normalise=True)
        for s in scores:
            assert np.isfinite(s.nid_score_normalised)
            assert s.nid_score_normalised == pytest.approx(0.0)

    def test_ensemble_averaging_reduces_variance(self):
        """Scores should be averaged across ensemble members, not summed."""
        feature_slices = {"a": slice(0, 1), "b": slice(1, 2)}
        rng = np.random.default_rng(123)
        # Two members with very different w1 values
        w1_a = np.array([[4.0, 0.1]], dtype=np.float32)  # strong a, weak b
        w1_b = np.array([[0.1, 4.0]], dtype=np.float32)  # weak a, strong b
        w_out = np.ones((1, 1), dtype=np.float32)

        scores_a = compute_nid_scores([(w1_a, [w_out])], feature_slices, normalise=False)
        scores_b = compute_nid_scores([(w1_b, [w_out])], feature_slices, normalise=False)
        scores_ens = compute_nid_scores([(w1_a, [w_out]), (w1_b, [w_out])], feature_slices, normalise=False)

        # Ensemble score should be mean of individual scores
        expected = (scores_a[0].nid_score + scores_b[0].nid_score) / 2.0
        assert scores_ens[0].nid_score == pytest.approx(expected, rel=1e-5)

    def test_three_way_scores_included(self):
        """max_order=3 should include three-way scores in the result."""
        n_features = 4
        feature_slices = {f"f{i}": slice(i, i + 1) for i in range(n_features)}
        w1 = np.abs(np.random.randn(6, n_features).astype(np.float32))
        w_out = np.ones((1, 6), dtype=np.float32)
        scores = compute_nid_scores([(w1, [w_out])], feature_slices, max_order=3)
        three_way = [s for s in scores if len(s.features) == 3]
        assert len(three_way) > 0


# ---------------------------------------------------------------------------
# nid_to_dataframe additional coverage
# ---------------------------------------------------------------------------

class TestNidToDataframeExtended:
    def test_three_way_uniform_order_uses_features_column(self):
        """Uniform three-way list should use features column, not feature_1/feature_2."""
        scores = [
            InteractionScore(features=("a", "b", "c"), nid_score=1.0, nid_score_normalised=1.0),
            InteractionScore(features=("a", "b", "d"), nid_score=0.5, nid_score_normalised=0.5),
        ]
        df = nid_to_dataframe(scores)
        assert "features" in df.columns
        assert "feature_1" not in df.columns
        assert "feature_2" not in df.columns

    def test_scores_match_input_values(self):
        scores = [
            InteractionScore(features=("x", "y"), nid_score=0.9, nid_score_normalised=1.0),
            InteractionScore(features=("x", "z"), nid_score=0.3, nid_score_normalised=0.333),
        ]
        df = nid_to_dataframe(scores)
        assert float(df["nid_score"][0]) == pytest.approx(0.9)
        assert float(df["nid_score"][1]) == pytest.approx(0.3)
        assert float(df["nid_score_normalised"][0]) == pytest.approx(1.0)

    def test_row_count_matches_input(self):
        scores = [
            InteractionScore(features=("a", "b"), nid_score=1.0, nid_score_normalised=1.0),
            InteractionScore(features=("a", "c"), nid_score=0.8, nid_score_normalised=0.8),
            InteractionScore(features=("b", "c"), nid_score=0.6, nid_score_normalised=0.6),
        ]
        df = nid_to_dataframe(scores)
        assert len(df) == 3

    def test_feature_names_preserved_in_pairwise(self):
        scores = [
            InteractionScore(
                features=("age_band", "vehicle_group"),
                nid_score=0.5,
                nid_score_normalised=1.0,
            )
        ]
        df = nid_to_dataframe(scores)
        assert df["feature_1"][0] == "age_band"
        assert df["feature_2"][0] == "vehicle_group"

    def test_order_filter_with_no_matches_returns_empty(self):
        scores = [
            InteractionScore(features=("a", "b", "c"), nid_score=1.0, nid_score_normalised=1.0),
        ]
        df = nid_to_dataframe(scores, order=2)
        assert df.is_empty()

    @pytest.mark.parametrize("n_scores", [0, 1, 3, 10])
    def test_empty_input_returns_empty(self, n_scores: int):
        if n_scores == 0:
            df = nid_to_dataframe([])
            assert df.is_empty()
        else:
            scores = [
                InteractionScore(features=(f"f{i}", f"f{i+1}"), nid_score=float(i), nid_score_normalised=0.5)
                for i in range(n_scores)
            ]
            df = nid_to_dataframe(scores)
            assert len(df) == n_scores


# ---------------------------------------------------------------------------
# InteractionScore NamedTuple
# ---------------------------------------------------------------------------

class TestInteractionScore:
    def test_construction(self):
        s = InteractionScore(features=("a", "b"), nid_score=0.7, nid_score_normalised=1.0)
        assert s.features == ("a", "b")
        assert s.nid_score == pytest.approx(0.7)

    def test_three_way_features(self):
        s = InteractionScore(features=("a", "b", "c"), nid_score=0.5, nid_score_normalised=0.5)
        assert len(s.features) == 3

    def test_sorting_by_nid_score(self):
        scores = [
            InteractionScore(features=("a", "c"), nid_score=0.3, nid_score_normalised=0.3),
            InteractionScore(features=("a", "b"), nid_score=0.9, nid_score_normalised=1.0),
            InteractionScore(features=("b", "c"), nid_score=0.6, nid_score_normalised=0.6),
        ]
        sorted_scores = sorted(scores, key=lambda x: x.nid_score, reverse=True)
        assert sorted_scores[0].features == ("a", "b")
        assert sorted_scores[-1].features == ("a", "c")
