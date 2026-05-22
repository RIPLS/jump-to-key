import pytest
from dino_jump.detector import JumpDetector, LandingEvent, TakeoffEvent


def test_starts_grounded_and_emits_nothing():
    det = JumpDetector(baseline=200.0, threshold=170.0)
    assert det.update(hip_y=200.0, t=0.0) is None
    assert det.state == "GROUNDED"


def test_emits_takeoff_when_crossing_threshold():
    det = JumpDetector(baseline=200.0, threshold=170.0)
    assert det.update(hip_y=200.0, t=0.0) is None
    event = det.update(hip_y=160.0, t=0.05)
    assert isinstance(event, TakeoffEvent)
    assert det.state == "AIRBORNE"


def test_emits_landing_event_with_airborne_duration():
    det = JumpDetector(baseline=200.0, threshold=170.0)
    det.update(hip_y=200.0, t=0.0)
    det.update(hip_y=160.0, t=0.10)   # takeoff
    det.update(hip_y=150.0, t=0.20)   # peak
    event = det.update(hip_y=200.0, t=0.40)  # landed
    assert isinstance(event, LandingEvent)
    assert event.airborne_seconds == pytest.approx(0.30, abs=1e-6)
    assert det.state == "GROUNDED"


def test_hysteresis_requires_clearing_baseline_minus_margin_to_land():
    det = JumpDetector(baseline=200.0, threshold=170.0, hysteresis=10.0)
    det.update(hip_y=200.0, t=0.0)
    det.update(hip_y=160.0, t=0.05)   # takeoff
    # 180 is below baseline-hysteresis (190); still airborne
    assert det.update(hip_y=180.0, t=0.10) is None
    assert det.state == "AIRBORNE"
    # 195 is above 190; lands now
    event = det.update(hip_y=195.0, t=0.15)
    assert isinstance(event, LandingEvent)
    assert det.state == "GROUNDED"


def test_multiple_airborne_frames_emit_nothing_until_landing():
    det = JumpDetector(baseline=200.0, threshold=170.0)
    det.update(hip_y=200.0, t=0.0)
    det.update(hip_y=160.0, t=0.10)   # takeoff
    assert det.update(hip_y=155.0, t=0.15) is None
    assert det.update(hip_y=150.0, t=0.20) is None
    assert det.state == "AIRBORNE"


def test_no_event_for_noise_that_never_crosses_threshold():
    det = JumpDetector(baseline=200.0, threshold=170.0)
    for i, y in enumerate([200, 198, 202, 199, 201, 200]):
        assert det.update(hip_y=float(y), t=i * 0.033) is None
    assert det.state == "GROUNDED"
