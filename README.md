# Dino Jump Cam

Play the Chrome dinosaur game by jumping in front of your webcam.

## Setup

```bash
git clone <repo-url> dino-jump-cam
cd dino-jump-cam
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Grant **Accessibility** and **Camera** permissions to your terminal app
(System Settings → Privacy & Security).

## Run

```bash
PYTHONPATH=src python -m dino_jump.main
```

Stand 1.5–2 m from the camera, stay still for 2 seconds to calibrate, then jump.

### Hotkeys (focus the preview window first)
- `p` — toggle paused. While paused, jumps are still detected and shown on
  screen, but no spacebar event is sent.
- `r` — recalibrate (use this if you change your standing position).
- `q` — quit.

## Test

```bash
PYTHONPATH=src pytest -v
```
