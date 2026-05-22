import pytest
from dino_jump.calibrator import Calibrator


def test_calibrator_not_ready_until_enough_samples():
    cal = Calibrator(required_samples=5)
    for _ in range(4):
        cal.add_sample(100.0)
    assert not cal.is_ready()


def test_calibrator_ready_at_required_samples():
    cal = Calibrator(required_samples=5)
    for _ in range(5):
        cal.add_sample(100.0)
    assert cal.is_ready()


def test_calibrator_computes_baseline_as_mean():
    cal = Calibrator(required_samples=4)
    for v in [100.0, 102.0, 98.0, 100.0]:
        cal.add_sample(v)
    assert cal.baseline == pytest.approx(100.0)


def test_calibrator_threshold_is_baseline_minus_k_std():
    # population std of [100, 102, 98, 100] ≈ 1.4142
    cal = Calibrator(required_samples=4, k=3.0, min_threshold_gap=0.0)
    for v in [100.0, 102.0, 98.0, 100.0]:
        cal.add_sample(v)
    assert cal.threshold == pytest.approx(100.0 - 3.0 * 1.4142135, abs=1e-3)


def test_calibrator_enforces_minimum_threshold_gap():
    # If std is tiny, threshold would equal baseline; min_gap forces separation.
    cal = Calibrator(required_samples=4, k=3.0, min_threshold_gap=10.0)
    for v in [100.0, 100.0, 100.0, 100.0]:
        cal.add_sample(v)
    assert cal.threshold == pytest.approx(90.0)


def test_baseline_raises_if_not_ready():
    cal = Calibrator(required_samples=5)
    with pytest.raises(RuntimeError):
        _ = cal.baseline


def test_progress_returns_sample_counts():
    cal = Calibrator(required_samples=10)
    cal.add_sample(100.0)
    cal.add_sample(100.0)
    assert cal.progress() == (2, 10)
