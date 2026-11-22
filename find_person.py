#!/usr/bin/env python3
"""
Find people (or other common objects) with the OAK-D Lite using an AI model.

The AI model (YOLOv6-Nano, trained on 80 everyday object types) runs ON THE
CAMERA'S OWN CHIP, so the Jetson's CPU and GPU stay free. The camera also
measures how far away each detection is, using its stereo depth.

USAGE
    python3 find_person.py                         # look for people
    python3 find_person.py --label bottle          # look for bottles instead
    python3 find_person.py --label person,backpack # several types at once
    python3 find_person.py --list-labels           # show every type it knows

WHAT IT PRINTS
    *** PERSON FOUND ***  1 in view | closest: person 3.4 m away, 12 deg right (confidence 87%)
    ... then a position update every second while the target stays in view ...
    person lost - searching...

  The direction (degrees right/left of the camera's centerline) and distance
  are the numbers a flight script would later use to turn toward the target.

NOTES
  * The first run downloads the AI model from Luxonis, so the Jetson needs
    internet once. It is cached afterwards.
  * It finds ANY person - it cannot tell one individual from another.
  * Distance comes from stereo depth: good to roughly 5-6 m, rough beyond that.
    It shows "distance unknown" when the target is closer than ~35 cm or depth fails.
  * Needs a screen for the live window; use --no-display over SSH.

Requires DepthAI v3:   python3 -m pip install -U depthai opencv-python numpy
"""

import argparse
import difflib
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import depthai as dai

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

# ---------------------------------------------------------------- settings
FOUND_NEEDED = 3          # consecutive detections before announcing FOUND
LOST_AFTER_S = 1.0        # no detection for this long => announce LOST
REPORT_EVERY_S = 1.0      # position update interval while target is in view
STATUS_EVERY_S = 5.0      # "still searching" message interval
DISPLAY_SCALE = 2.0       # the AI's input picture is small; enlarge it for the window
WINDOW = "Target finder  (q = quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------ detections
@dataclass
class Hit:
    name: str
    confidence: float
    box: Tuple[float, float, float, float]   # normalized xmin, ymin, xmax, ymax
    x_mm: float                              # right of the camera axis
    y_mm: float                              # below the camera axis
    z_mm: float                              # forward distance (0 = unknown)

    @property
    def has_depth(self) -> bool:
        return self.z_mm > 0

    @property
    def distance_m(self) -> Optional[float]:
        if not self.has_depth:
            return None
        return math.sqrt(self.x_mm ** 2 + self.y_mm ** 2 + self.z_mm ** 2) / 1000.0

    @property
    def bearing_deg(self) -> Optional[float]:
        """Degrees right (+) or left (-) of the camera's centerline."""
        if not self.has_depth:
            return None
        return math.degrees(math.atan2(self.x_mm, self.z_mm))

    def where(self) -> str:
        b = self.bearing_deg
        if b is not None:
            return "straight ahead" if abs(b) < 5 else f"{abs(b):.0f} deg {'right' if b > 0 else 'left'}"
        cx = (self.box[0] + self.box[2]) / 2
        off = (cx - 0.5) * 200
        return "centered" if abs(off) < 15 else f"{abs(off):.0f}% {'right' if off > 0 else 'left'} of center"

    def describe(self) -> str:
        d = self.distance_m
        dist = f"{d:.1f} m away" if d is not None else "distance unknown"
        return f"{self.name} {dist}, {self.where()} (confidence {self.confidence:.0%})"


def extract_hits(detections, labels, wanted) -> Tuple[List[Hit], List[Hit]]:
    """Split detections into (wanted targets, everything else). Closest target first."""
    targets, others = [], []
    for d in detections:
        if labels and 0 <= d.label < len(labels):
            name = labels[d.label]
        else:
            name = d.labelName or str(d.label)
        sc = d.spatialCoordinates
        h = Hit(name, float(d.confidence), (d.xmin, d.ymin, d.xmax, d.ymax),
                float(sc.x), float(sc.y), float(sc.z))
        (targets if name in wanted else others).append(h)
    # targets with a valid distance first (nearest first), then the rest by confidence
    targets.sort(key=lambda h: (not h.has_depth, h.z_mm if h.has_depth else 0.0, -h.confidence))
    return targets, others


class FoundTracker:
    """Debounce so one missed frame doesn't flip FOUND/LOST."""

    def __init__(self):
        self.found = False
        self.hits = 0
        self.last_seen = 0.0

    def update(self, detected: bool, now: float) -> Optional[str]:
        if detected:
            self.hits += 1
            self.last_seen = now
        else:
            self.hits = 0
        if not self.found and self.hits >= FOUND_NEEDED:
            self.found = True
            return "FOUND"
        if self.found and now - self.last_seen > LOST_AFTER_S:
            self.found = False
            return "LOST"
        return None


# ------------------------------------------------------------------ drawing
def put_text(img, text, org, scale=0.7, thickness=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), thickness)


def draw_view(frame, targets: List[Hit], others: List[Hit], tracker: FoundTracker,
              banner_name: str, show_all: bool):
    view = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
    h, w = view.shape[:2]

    def rect(hit):
        x0, y0, x1, y1 = hit.box
        return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)

    if show_all:
        for o in others:
            x0, y0, x1, y1 = rect(o)
            cv2.rectangle(view, (x0, y0), (x1, y1), (150, 150, 150), 1)
            put_text(view, o.name, (x0 + 4, y0 + 18), 0.5, 1)
    for i, t in enumerate(targets):
        x0, y0, x1, y1 = rect(t)
        color = (0, 255, 0) if i == 0 else (0, 200, 200)
        cv2.rectangle(view, (x0, y0), (x1, y1), color, 3)
        d = t.distance_m
        put_text(view, f"{t.name} {t.confidence:.0%}" + (f"  {d:.1f} m" if d else ""),
                 (x0 + 4, max(20, y0 - 8)), 0.7)

    banner_color = (0, 170, 0) if tracker.found else (60, 60, 60)
    cv2.rectangle(view, (0, 0), (w, 50), banner_color, -1)
    if tracker.found and targets:
        extra = f" (+{len(targets) - 1} more)" if len(targets) > 1 else ""
        put_text(view, f"{banner_name.upper()} FOUND  {targets[0].where()}{extra}", (10, 35), 0.9)
    else:
        put_text(view, f"searching for {banner_name}...", (10, 35), 0.9)
    return view


# --------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Find people/objects with AI on the OAK-D Lite.")
    p.add_argument("--label", default="person",
                   help="what to look for, e.g. person, bottle, 'person,backpack' (default: person)")
    p.add_argument("--conf", type=float, default=0.5,
                   help="minimum confidence 0-1 (lower = more sensitive, more false alarms)")
    p.add_argument("--fps", type=float, default=15, help="camera/AI frame rate")
    p.add_argument("--model", default="yolov6-nano",
                   help="Luxonis model zoo name (default yolov6-nano; must be a COCO model)")
    p.add_argument("--show-all", action="store_true",
                   help="also outline every other object the AI sees (gray)")
    p.add_argument("--no-display", action="store_true", help="no window; print status only")
    p.add_argument("--list-labels", action="store_true", help="list everything the model can detect")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.list_labels:
        print("The model can detect these 80 types:\n")
        print(", ".join(COCO_CLASSES))
        return

    wanted = {s.strip().lower() for s in args.label.split(",") if s.strip()}
    for w in sorted(wanted):
        if w not in COCO_CLASSES:
            close = difflib.get_close_matches(w, COCO_CLASSES, n=3, cutoff=0.5)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            sys.exit(f"'{w}' isn't something the model knows.{hint}  (see --list-labels)")
    banner_name = " / ".join(sorted(wanted))

    if int(dai.__version__.split(".")[0]) < 3:
        sys.exit("This script needs DepthAI v3. Run: python3 -m pip install -U depthai")

    show = (not args.no_display) and bool(os.environ.get("DISPLAY"))
    if not show:
        log("No display - printing status to the terminal instead of a window")

    log("Starting camera + AI model (the first run downloads the model - needs internet)...")
    with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left.requestOutput((640, 400), fps=args.fps),
            right.requestOutput((640, 400), fps=args.fps),
            presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS,
        )
        nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
            cam, stereo, dai.NNModelDescription(args.model), fps=args.fps
        )
        nn.input.setBlocking(False)
        nn.setConfidenceThreshold(args.conf)
        nn.setBoundingBoxScaleFactor(0.5)    # measure depth on the middle of the box, not the edges
        nn.setDepthLowerThreshold(100)       # ignore depth closer than 0.1 m
        nn.setDepthUpperThreshold(10000)     # ignore depth beyond 10 m
        labels = nn.getClasses() or COCO_CLASSES
        q_det = nn.out.createOutputQueue(maxSize=1, blocking=False)
        q_rgb = nn.passthrough.createOutputQueue(maxSize=1, blocking=False)

        pipeline.start()
        log(f"Running. Looking for: {banner_name}   (confidence >= {args.conf:.0%})")

        tracker = FoundTracker()
        last_status = last_report = time.monotonic()
        try:
            while pipeline.isRunning():
                det_msg = q_det.get()
                frame = q_rgb.get().getCvFrame()
                targets, others = extract_hits(det_msg.detections, labels, wanted)
                now = time.monotonic()
                event = tracker.update(bool(targets), now)

                if event == "FOUND":
                    log(f"*** {banner_name.upper()} FOUND ***  {len(targets)} in view | "
                        f"closest: {targets[0].describe()}")
                    last_report = now
                elif event == "LOST":
                    log(f"{banner_name} lost - searching...")
                    last_status = now
                elif tracker.found and targets and now - last_report > REPORT_EVERY_S:
                    log(f"in view: {len(targets)} | closest: {targets[0].describe()}")
                    last_report = now
                elif not tracker.found and now - last_status > STATUS_EVERY_S:
                    log("still searching...")
                    last_status = now

                if show:
                    cv2.imshow(WINDOW, draw_view(frame, targets, others, tracker,
                                                 banner_name, args.show_all))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            log("Ctrl+C - exiting")
        finally:
            if show:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()