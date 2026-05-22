import time

import cv2

from dino_jump.camera import open_camera
from dino_jump.calibrator import Calibrator
from dino_jump.detector import JumpDetector, LandingEvent, TakeoffEvent
from dino_jump.keypress import press_space_down, press_space_up
from dino_jump.pose import PoseTracker


CALIBRATION_SAMPLES = 60          # ~2 s at 30 FPS
JUMP_FLASH_SECONDS = 0.35         # how long the "JUMP!" overlay stays on
MAX_HOLD_SECONDS = 1.0            # safety: never hold space longer than this


def _new_calibrator() -> Calibrator:
    return Calibrator(required_samples=CALIBRATION_SAMPLES)


def run() -> None:
    cap = open_camera()
    tracker = PoseTracker()
    calibrator = _new_calibrator()
    detector: JumpDetector | None = None
    paused = False
    space_held = False
    held_since: float | None = None
    last_jump_at: float | None = None
    last_jump_label = ""

    def safe_release(reason: str = "") -> None:
        nonlocal space_held, held_since
        if space_held:
            press_space_up()
            space_held = False
            held_since = None
            if reason:
                print(f"Released space ({reason})")

    print("Stand still for ~2 seconds to calibrate.")
    print("Hotkeys (focus preview window): p=pause  r=recalibrate  q=quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.flip(frame, 1)  # mirror for natural feedback
            t = time.monotonic()

            # Safety: if we've been holding space too long (e.g. pose lost
            # mid-jump and never came back), force-release.
            if space_held and held_since is not None and (t - held_since) > MAX_HOLD_SECONDS:
                safe_release(reason="timeout")

            pose_result = tracker.process(frame)
            tracker.draw(frame, pose_result.raw_landmarks)

            y = pose_result.hip_y
            if y is None:
                status = "NO POSE"
            elif detector is None:
                calibrator.add_sample(y)
                done, total = calibrator.progress()
                status = f"CALIBRATING {done}/{total}"
                if calibrator.is_ready():
                    detector = JumpDetector(
                        baseline=calibrator.baseline,
                        threshold=calibrator.threshold,
                    )
                    print(
                        f"Calibrated. baseline={detector.baseline:.1f} "
                        f"threshold={detector.threshold:.1f}"
                    )
            else:
                event = detector.update(hip_y=y, t=t)
                status = detector.state
                if isinstance(event, TakeoffEvent):
                    last_jump_at = t
                    last_jump_label = f"TAKEOFF{' (suppressed)' if paused else ''}"
                    print(last_jump_label)
                    if not paused:
                        press_space_down()
                        space_held = True
                        held_since = t
                elif isinstance(event, LandingEvent):
                    last_jump_label = (
                        f"LANDING airborne={event.airborne_seconds*1000:.0f}ms"
                        f"{' (suppressed)' if paused else ''}"
                    )
                    print(last_jump_label)
                    safe_release()

            # HUD: baseline + threshold lines
            if detector is not None:
                cv2.line(
                    frame,
                    (0, int(detector.baseline)),
                    (frame.shape[1], int(detector.baseline)),
                    (0, 255, 0), 1,
                )
                cv2.line(
                    frame,
                    (0, int(detector.threshold)),
                    (frame.shape[1], int(detector.threshold)),
                    (0, 0, 255), 1,
                )

            # HUD: status bar
            mode = "PAUSED" if paused else "ACTIVE"
            mode_color = (0, 200, 255) if paused else (0, 255, 0)
            held_marker = "  [SPACE]" if space_held else ""
            cv2.putText(
                frame, f"{mode} | {status}{held_marker}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2,
            )
            cv2.putText(
                frame, "p=pause  r=recalibrate  q=quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )

            # HUD: jump flash overlay
            if last_jump_at is not None and (t - last_jump_at) < JUMP_FLASH_SECONDS:
                cv2.putText(
                    frame, "JUMP!", (frame.shape[1] // 2 - 80, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4,
                )
                cv2.putText(
                    frame, last_jump_label,
                    (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                )

            cv2.imshow("dino-jump-cam", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                paused = not paused
                if paused:
                    safe_release(reason="paused")
                print(f"{'PAUSED' if paused else 'ACTIVE'}")
            elif key == ord("r"):
                print("Recalibrating — stand still.")
                safe_release(reason="recalibrate")
                calibrator = _new_calibrator()
                detector = None
                last_jump_at = None
    finally:
        safe_release(reason="shutdown")
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    run()
