import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from tora.eval import metrics as M
from tora.models.baselines.qwen3guard_stream import CONTROVERSIAL as C
from tora.models.baselines.qwen3guard_stream import SAFE as S
from tora.models.baselines.qwen3guard_stream import UNSAFE as U
from tora.models.baselines.qwen3guard_stream import official_response_decision


def test_prf_at():
    out = M.prf_at([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (out["precision"], out["recall"], out["f1"], out["accuracy"]) == (0.5, 0.5, 0.5, 0.5)


def test_best_f1_threshold_is_optimal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    s = y * 0.5 + rng.normal(0, 0.5, 300)
    thr = M.best_f1_threshold(y, s)
    brute = max(M.prf_at(y, s, t)["f1"] for t in s)
    assert M.prf_at(y, s, thr)["f1"] == pytest.approx(brute)


def test_classification_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1, 500)
    out = M.classification_metrics(y, s, threshold=0.5)
    assert out["auroc"] == pytest.approx(roc_auc_score(y, s))
    assert {"auprc", "fpr@95tpr", "f1"} <= out.keys()


def test_fpr_at_tpr():
    # Positives at 0.9, 0.8, 0.3 -> TPR 1.0 needs threshold 0.3, which also flags the negative at 0.5.
    assert M.fpr_at_tpr([1, 1, 1, 0, 0], [0.9, 0.8, 0.3, 0.5, 0.1], tpr=0.95) == 0.5


@pytest.mark.parametrize("target", [0.0, 0.01, 0.05, 0.2, 1.0])
def test_threshold_at_fpr(target):
    rng = np.random.default_rng(2)
    neg = np.round(rng.random(200), 2)  # ties on purpose
    thr = M.threshold_at_fpr(neg, target)
    assert (neg >= thr).mean() <= target + 1e-12
    if target < 1.0:  # no lower threshold among the scores keeps FPR within target
        lower = neg[neg < thr]
        if lower.size:
            assert (neg >= lower.max()).mean() > target


def test_consecutive_rule():
    s = np.array([0.9, 0.1, 0.9, 0.9, 0.2])
    assert M.first_flag(s, 0.5, consecutive=1) == 0
    assert M.first_flag(s, 0.5, consecutive=2) == 3
    assert M.first_flag(s, 0.95, consecutive=1) == -1
    rng = np.random.default_rng(3)
    for _ in range(200):  # response_score is consistent with the streaming rule at every threshold
        x, thr, k = rng.random(rng.integers(1, 12)), rng.random(), int(rng.integers(1, 4))
        assert (M.response_score(x, k) >= thr) == (M.first_flag(x, thr, k) >= 0)


def test_flag_metrics_bins():
    spans = [(10, 20)] * 6
    flags = [5, 10, 20, 21, 140, -1]  # 140 is 120 tokens after the span end
    out = M.flag_metrics(flags, spans)
    assert out["bins"] == {"ahead": 1, "hit": 2, "1-32": 1, "33-64": 0, "65-128": 1, "129-256": 0, ">256": 0, "missed": 1}
    assert out["detection_rate"] == pytest.approx(5 / 6)
    assert out["within_128_rate"] == pytest.approx(5 / 6)  # flags <= end + 128, ahead included (Qwen's definition)
    assert out["exact_hit_rate"] == pytest.approx(2 / 6)


def test_streaming_metrics_fpr_on_safe():
    unsafe = [np.array([0.1, 0.9, 0.9])]
    safe = [np.array([0.1, 0.2]), np.array([0.95, 0.95])]
    out = M.streaming_metrics(unsafe, [(1, 2)], threshold=0.5, consecutive=2, safe_scores=safe)
    assert out["bins"]["hit"] == 1 and out["fpr_safe"] == 0.5


@pytest.mark.parametrize(
    "levels, expected",
    [
        ([S, S, S], (S, -1)),
        ([U, C, U, C], (S, -1)),  # mixed pairs do not count
        ([C, C, U, U], (U, 3)),  # Unsafe pairs are checked first
        ([S, C, C], (C, 2)),
        ([U, U], (U, 1)),
        ([U], (S, -1)),
    ],
)
def test_qwen3guard_official_rule(levels, expected):
    assert official_response_decision(levels) == expected
