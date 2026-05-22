from dataclasses import dataclass

import mediapipe as mp
import numpy as np

_mp_pose = mp.solutions.pose
_mp_draw = mp.solutions.drawing_utils
_mp_styles = mp.solutions.drawing_styles


@dataclass
class PoseResult:
    hip_y: float | None       # average y pixel coord of L+R hip, or None
    raw_landmarks: object | None  # MediaPipe NormalizedLandmarkList or None


class PoseTracker:
    """Wraps MediaPipe Pose."""

    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self):
        self._pose = _mp_pose.Pose(
            model_complexity=0,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        frame_rgb = frame_bgr[:, :, ::-1]
        result = self._pose.process(frame_rgb)
        if not result.pose_landmarks:
            return PoseResult(hip_y=None, raw_landmarks=None)

        lm = result.pose_landmarks.landmark
        left, right = lm[self.LEFT_HIP], lm[self.RIGHT_HIP]
        if left.visibility < 0.5 or right.visibility < 0.5:
            return PoseResult(hip_y=None, raw_landmarks=result.pose_landmarks)

        h = frame_bgr.shape[0]
        hip_y = ((left.y + right.y) / 2.0) * h
        return PoseResult(hip_y=hip_y, raw_landmarks=result.pose_landmarks)

    def draw(self, frame_bgr: np.ndarray, landmarks) -> None:
        """Draw the pose skeleton onto frame_bgr in place."""
        if landmarks is None:
            return
        _mp_draw.draw_landmarks(
            frame_bgr,
            landmarks,
            _mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=_mp_styles.get_default_pose_landmarks_style(),
        )

    def close(self) -> None:
        self._pose.close()
