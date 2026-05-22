import statistics


class Calibrator:
    """Collects hip_y samples while the user stands still and derives a
    baseline (mean) and a jump threshold (baseline - gap)."""

    def __init__(
        self,
        required_samples: int = 60,
        k: float = 3.0,
        min_threshold_gap: float = 15.0,
    ):
        self._required = required_samples
        self._k = k
        self._min_gap = min_threshold_gap
        self._samples: list[float] = []

    def add_sample(self, hip_y: float) -> None:
        if not self.is_ready():
            self._samples.append(hip_y)

    def is_ready(self) -> bool:
        return len(self._samples) >= self._required

    def progress(self) -> tuple[int, int]:
        return (len(self._samples), self._required)

    @property
    def baseline(self) -> float:
        if not self.is_ready():
            raise RuntimeError("Calibrator not ready yet")
        return statistics.fmean(self._samples)

    @property
    def threshold(self) -> float:
        if not self.is_ready():
            raise RuntimeError("Calibrator not ready yet")
        std = statistics.pstdev(self._samples)
        gap = max(self._k * std, self._min_gap)
        return self.baseline - gap
