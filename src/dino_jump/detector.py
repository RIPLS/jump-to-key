from dataclasses import dataclass


@dataclass(frozen=True)
class TakeoffEvent:
    pass


@dataclass(frozen=True)
class LandingEvent:
    airborne_seconds: float


class JumpDetector:
    """FSM watching hip_y over time. Emits a TakeoffEvent on the GROUNDED →
    AIRBORNE transition and a LandingEvent on the AIRBORNE → GROUNDED
    transition.

    Image coords: smaller y = higher in the frame. So 'airborne' means
    hip_y < threshold (where threshold = baseline - gap)."""

    def __init__(
        self,
        baseline: float,
        threshold: float,
        hysteresis: float = 10.0,
    ):
        self.baseline = baseline
        self.threshold = threshold
        self._land_y = baseline - hysteresis
        self.state = "GROUNDED"
        self._takeoff_t: float | None = None

    def update(self, hip_y: float, t: float) -> TakeoffEvent | LandingEvent | None:
        if self.state == "GROUNDED":
            if hip_y < self.threshold:
                self.state = "AIRBORNE"
                self._takeoff_t = t
                return TakeoffEvent()
            return None

        # AIRBORNE
        if hip_y >= self._land_y:
            airborne = t - (self._takeoff_t or t)
            self.state = "GROUNDED"
            self._takeoff_t = None
            return LandingEvent(airborne_seconds=airborne)
        return None
