#!/usr/bin/env python3
"""
Cup Anemometer — ROBOFLOW MODEL version (gate / lap timing).

Same lap-timer math as the color version, but detection comes from your
custom-trained Roboflow object-detection model running LOCALLY on your Mac
(no per-frame internet calls). Each time the detected object (re)appears,
that's a gate pass; wind speed comes from the time between passes:

    omega = 2*pi / (PASSES_PER_REV * T_pass)
    wind  = K_FACTOR * omega * ROTOR_RADIUS_M

Setup:
  pip install inference opencv-python numpy        (Python >=3.9, <3.13)
  export ROBOFLOW_API_KEY="your_private_api_key"   (from app.roboflow.com settings)

Run:
  python3 cup_anemometer_roboflow.py --model-id your-project/1
  python3 cup_anemometer_roboflow.py --model-id your-project/1 --class-name marker
  python3 cup_anemometer_roboflow.py --model-id your-project/1 \
          --class-name cup --passes-per-rev 3      (if you labeled the cups:
                                                    3 cups pass per revolution)

Gate modes:
  Default = FULL-FRAME: a pass is counted each time the object appears after
  being absent. Use this when you labeled the single marked cup / marker.
  CLICK anywhere = spot gate at that point (press f to go back to full-frame).
  If you labeled ALL the cups (--passes-per-rev 3), use a SPOT gate — with
  three cups, one is almost always visible, so full-frame never re-arms.

Keys:  q quit   r reset laps   f toggle full-frame gate   +/- gate size

The first run downloads the model weights (needs internet once); after that
it runs offline. Inference speed is shown on screen — if effective fps is
low, lower --infer-width (e.g. 480) to trade resolution for speed.
"""

import argparse
import csv
import math
import os
import time
from collections import deque

import cv2
import numpy as np

# ----------------------------- physical / tuning constants -----------------------------

ROTOR_RADIUS_M = 0.12065     # meters, spindle axis -> cup center. MEASURE THIS.
K_FACTOR       = 2.8         # anemometer factor (wind speed / cup speed). CALIBRATE THIS.

GATE_RADIUS_PX  = 90         # spot-gate radius (adjust live with + / -)
REARM_FRAMES    = 2          # frames out of gate required to re-arm
MIN_PERIOD_S    = 0.06       # debounce floor between passes
MEDIAN_LAPS     = 5          # median over this many recent pass intervals
LOST_AFTER_LAPS = 4          # no pass for this many expected intervals -> NO SIGNAL
LOST_AFTER_SEC  = 6.0
CSV_PATH        = "wind_roboflow_log.csv"

# ------------------------------------ lap timer ------------------------------------------

class LapTimer:
    """Photo-gate pass timer with debounce, missed-pass folding, and timeout."""

    def __init__(self, passes_per_rev=1):
        self.ppr = passes_per_rev
        self.pass_times = deque(maxlen=60)
        self.periods = deque(maxlen=MEDIAN_LAPS)
        self.armed = True
        self.frames_outside = 0
        self.flash_until = 0.0

    def reset(self):
        self.__init__(self.ppr)

    def update(self, now, in_gate):
        registered = False
        if in_gate:
            self.frames_outside = 0
            if self.armed:
                self.armed = False
                if self.pass_times:
                    p = now - self.pass_times[-1]
                    if p >= MIN_PERIOD_S:
                        med = self.median_period()
                        if med and p > 1.75 * med:     # fold missed passes
                            p = p / max(1, round(p / med))
                        self.periods.append(p)
                        registered = True
                self.pass_times.append(now)
                self.flash_until = now + 0.15
        else:
            self.frames_outside += 1
            if self.frames_outside >= REARM_FRAMES:
                self.armed = True
        return registered

    def median_period(self):
        return float(np.median(self.periods)) if self.periods else None

    def status(self, now):
        med = self.median_period()
        if med is None:
            state = "WAITING FOR PASSES" if self.pass_times else "WAITING FOR FIRST PASS"
            return state, 0.0, 0.0, 0.0
        age = now - self.pass_times[-1]
        if age > min(LOST_AFTER_LAPS * med, LOST_AFTER_SEC):
            return "NO SIGNAL", 0.0, 0.0, 0.0
        rev_period = med * self.ppr
        omega = 2.0 * math.pi / rev_period
        wind = K_FACTOR * omega * ROTOR_RADIUS_M
        return "TIMING", med, omega, wind


# --------------------------------------- main --------------------------------------------

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True,
                    help="Roboflow model id, e.g. cup-anemometer/1")
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY"),
                    help="Roboflow private API key (or set ROBOFLOW_API_KEY)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.4,
                    help="confidence threshold (default 0.4)")
    ap.add_argument("--class-name", default=None,
                    help="only count detections of this class (default: any)")
    ap.add_argument("--passes-per-rev", type=int, default=1,
                    help="gate passes per revolution: 1 for a single marker, "
                         "3 if you labeled all three cups")
    ap.add_argument("--infer-width", type=int, default=640,
                    help="downscale frames to this width before inference "
                         "(smaller = faster, default 640)")
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("No API key. Run:  export ROBOFLOW_API_KEY=\"...\"  "
                         "(find it at app.roboflow.com > Settings > API Keys)")

    print("Loading model (first run downloads weights, needs internet once)…")
    from inference import get_model            # heavy import, do it after arg checks
    model = get_model(model_id=args.model_id, api_key=args.api_key)
    print("Model loaded.")

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}.")

    timer = LapTimer(args.passes_per_rev)
    gate = None
    gate_r = GATE_RADIUS_PX
    full_frame = True
    fps_clock = deque(maxlen=30)
    fps, infer_ms = 0.0, 0.0

    log = open(CSV_PATH, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["t_unix", "event", "pass_interval_s", "rpm",
                     "wind_mps", "wind_mph"])

    win = "cup anemometer (roboflow)"
    cv2.namedWindow(win)
    clicks = []
    cv2.setMouseCallback(win, lambda ev, x, y, flags, param:
                         clicks.append((x, y)) if ev == cv2.EVENT_LBUTTONDOWN else None)

    print("FULL-FRAME gate is ON - spin it! Click to place a spot gate instead.")
    print("Keys: q quit  r reset  f toggle full-frame  +/- gate size")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            now = time.time()
            fps_clock.append(now)
            if len(fps_clock) > 1:
                fps = (len(fps_clock) - 1) / (fps_clock[-1] - fps_clock[0] + 1e-9)

            if clicks:
                x, y = clicks.pop()
                gate = (x, y)
                full_frame = False
                timer.reset()
                print(f"Spot gate at ({x},{y}), r={gate_r}px. f = full-frame.")

            # ------------------------------ inference ------------------------------
            h, w = frame.shape[:2]
            if args.infer_width and w > args.infer_width:
                scale = args.infer_width / w
                small = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                scale, small = 1.0, frame

            t0 = time.time()
            result = model.infer(small, confidence=args.conf)[0]
            infer_ms = (time.time() - t0) * 1000.0

            dets = []   # (x, y, conf, label) in full-frame coordinates
            for p in result.predictions:
                label = getattr(p, "class_name", None) or getattr(p, "class_", "")
                if args.class_name and label != args.class_name:
                    continue
                dets.append((p.x / scale, p.y / scale, p.confidence, label))

            # ------------------------------ gate logic -----------------------------
            if full_frame:
                in_gate = len(dets) > 0
            else:
                in_gate = any(math.hypot(d[0] - gate[0], d[1] - gate[1]) <= gate_r
                              for d in dets) if gate else False
            registered = timer.update(now, in_gate)

            state, interval, omega, wind = timer.status(now)
            rpm = omega * 60.0 / (2.0 * math.pi)
            mph = wind * 2.23694
            if registered:
                writer.writerow([f"{now:.3f}", "pass", f"{interval:.4f}",
                                 f"{rpm:.2f}", f"{wind:.3f}", f"{mph:.3f}"])
                print(f"pass {interval:.3f}s   {rpm:6.1f} RPM   "
                      f"wind {wind:5.2f} m/s ({mph:5.2f} mph)")

            # ------------------------------- overlay -------------------------------
            vis = frame
            for (dx, dy, conf, label) in dets:
                cv2.circle(vis, (int(dx), int(dy)), 12, (0, 255, 0), 2)
                cv2.putText(vis, f"{label} {conf:.2f}",
                            (int(dx) + 15, int(dy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            if not full_frame and gate is not None:
                flash = now < timer.flash_until
                col = (0, 255, 0) if flash else (0, 255, 255)
                cv2.circle(vis, gate, gate_r, col, 3 if flash else 2)
            elif full_frame and now < timer.flash_until:
                cv2.rectangle(vis, (4, 4), (w - 5, h - 5), (0, 255, 0), 6)

            col = {"TIMING": (0, 255, 0), "NO SIGNAL": (0, 0, 255)}.get(
                state, (0, 200, 255))
            mode = "FULL-FRAME" if full_frame else "spot gate"
            eff_fps = min(fps, 1000.0 / max(infer_ms, 1e-3))
            lines = [
                f"{state} [{mode}] wind {wind:5.2f} m/s ({mph:5.2f} mph) "
                f"{rpm:6.1f} RPM",
                f"pass {interval:5.3f}s  passes {len(timer.pass_times)}  "
                f"infer {infer_ms:5.1f} ms  ~{eff_fps:4.1f} fps",
            ]
            if interval > 0 and interval * eff_fps < 6:
                lines.append("WARNING: passes too fast for inference speed - "
                              "lower --infer-width or slow the rotor")
            for i, txt in enumerate(lines):
                cv2.putText(vis, txt, (12, 32 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 5)
                cv2.putText(vis, txt, (12, 32 + 30 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            col if i == 0 else (255, 255, 255), 2)

            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                timer.reset()
                print("Lap data reset.")
            elif key == ord("f"):
                full_frame = not full_frame
                timer.reset()
                print(f"Full-frame gate: {'ON' if full_frame else 'OFF'}")
            elif key in (ord("+"), ord("=")):
                gate_r = min(400, gate_r + 10)
                print(f"Gate radius: {gate_r}px")
            elif key == ord("-"):
                gate_r = max(20, gate_r - 10)
                print(f"Gate radius: {gate_r}px")
    finally:
        log.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
