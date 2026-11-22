#!/usr/bin/env python3
"""
Find ONE specific target with the OAK-D Lite color camera.

STEP 1 - teach it what the target looks like
    python3 find_target.py --enroll
  A live window opens. Aim the camera at the target and press C to freeze the
  picture, drag a box tightly around the target, then press ENTER. The crop is
  saved as target.png. (Or skip this and use any photo of the target:
  --target my_photo.jpg)

STEP 2 - let it search
    python3 find_target.py
  It prints "TARGET FOUND" with where the target is in the picture, and
  "target lost" when it disappears. Nothing here touches the drone.

HOW IT WORKS
  Feature matching (SIFT): the script remembers distinctive details of the
  target (corners, lettering, pattern) and looks for the same details in each
  camera frame. It copes with the target being nearer/farther, rotated or
  seen at an angle.

  Good targets: anything flat and detailed - a poster, logo, sign, printed
                pattern, book cover, product label.
  Poor targets: plain or smooth objects (a white cup, a ball), shiny
                surfaces, or a target that is tiny in the picture.

Requires DepthAI v3:   python3 -m pip install -U depthai opencv-python numpy
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import depthai as dai

# ---------------------------------------------------------------- settings
FOUND_NEEDED = 3          # consecutive detections before announcing FOUND
LOST_NEEDED = 10          # consecutive misses before announcing LOST
RATIO_TEST = 0.75         # Lowe ratio test: lower = stricter matching
MIN_REF_KEYPOINTS = 20    # a target with fewer details than this is too plain
REF_MAX_SIDE = 480        # target image is shrunk to this many pixels (longest side)
STATUS_EVERY_S = 5.0      # "still searching" message interval
REPORT_EVERY_S = 1.0      # position report interval while target is in view
WINDOW = "Target finder  (q = quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- matching
@dataclass
class Detection:
    quad: np.ndarray            # 4x2 corners of the target as seen in the frame
    center: Tuple[float, float]
    inliers: int                # number of agreeing feature matches (match strength)


class TargetMatcher:
    def __init__(self, ref_bgr: np.ndarray, kind: str = "sift", min_inliers: int = 20):
        if kind == "sift" and hasattr(cv2, "SIFT_create"):
            self.kind = "sift"
            self.det = cv2.SIFT_create(nfeatures=1500)
            norm = cv2.NORM_L2
        else:
            if kind == "sift":
                log("SIFT not available in this OpenCV build - using ORB instead")
            self.kind = "orb"
            self.det = cv2.ORB_create(nfeatures=1500)
            norm = cv2.NORM_HAMMING
        self.matcher = cv2.BFMatcher(norm)
        self.min_inliers = min_inliers

        h, w = ref_bgr.shape[:2]
        longest = max(h, w)
        if longest > REF_MAX_SIDE:
            s = REF_MAX_SIDE / longest
            ref_bgr = cv2.resize(ref_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)
        self.kp_ref, self.des_ref = self.det.detectAndCompute(gray, None)
        self.n_ref = 0 if self.des_ref is None else len(self.kp_ref)
        if self.n_ref < MIN_REF_KEYPOINTS:
            raise ValueError(
                f"The target image only has {self.n_ref} usable details (need {MIN_REF_KEYPOINTS}+). "
                "It is too plain or blurry. Pick a more detailed target, or crop closer."
            )
        rh, rw = gray.shape
        self.ref_corners = np.float32([[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)

    def find(self, frame_bgr: np.ndarray) -> Optional[Detection]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self.det.detectAndCompute(gray, None)
        if des is None or len(kp) < 2:
            return None

        pairs = self.matcher.knnMatch(self.des_ref, des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO_TEST * p[1].distance]
        if len(good) < self.min_inliers:
            return None

        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return None
        inliers = int(mask.sum())
        if inliers < self.min_inliers:
            return None

        quad = cv2.perspectiveTransform(self.ref_corners, H).reshape(4, 2)
        if not self._plausible(quad, gray.shape):
            return None
        cx, cy = quad.mean(axis=0)
        return Detection(quad, (float(cx), float(cy)), inliers)

    @staticmethod
    def _plausible(quad: np.ndarray, shape) -> bool:
        """Reject warped/garbage outlines: must be a convex shape of sensible size."""
        pts = quad.astype(np.float32)
        if not cv2.isContourConvex(pts.astype(np.int32)):
            return False
        area = cv2.contourArea(pts)
        fh, fw = shape[:2]
        return 400 < area < 1.5 * fw * fh


class FoundTracker:
    """Debounce: announce FOUND after a few hits in a row, LOST after a longer run of misses."""

    def __init__(self):
        self.found = False
        self.hits = 0
        self.misses = 0

    def update(self, detected: bool) -> Optional[str]:
        if detected:
            self.hits += 1
            self.misses = 0
        else:
            self.misses += 1
            self.hits = 0
        if not self.found and self.hits >= FOUND_NEEDED:
            self.found = True
            return "FOUND"
        if self.found and self.misses >= LOST_NEEDED:
            self.found = False
            return "LOST"
        return None


def where(center_x: float, frame_w: int) -> str:
    off = (center_x - frame_w / 2) / (frame_w / 2) * 100
    if abs(off) < 15:
        return "centered"
    return f"{abs(off):.0f}% {'right' if off > 0 else 'left'} of center"


# ------------------------------------------------------------------ drawing
def put_text(img, text, org, scale=0.8, thickness=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), thickness)


def draw_view(frame, det: Optional[Detection], tracker: FoundTracker, fps: float):
    view = frame.copy()
    if det is not None:
        cv2.polylines(view, [det.quad.astype(np.int32)], True, (0, 255, 0), 3)
        put_text(view, f"match strength {det.inliers}", (int(det.quad[0][0]), max(25, int(det.quad[0][1]) - 10)), 0.7)
    color = (0, 170, 0) if tracker.found else (60, 60, 60)
    cv2.rectangle(view, (0, 0), (view.shape[1], 50), color, -1)
    put_text(view, "TARGET FOUND" if tracker.found else "searching...", (10, 35), 1.0)
    put_text(view, f"{fps:.0f} fps", (view.shape[1] - 130, 35), 0.8)
    return view


# ------------------------------------------------------------------- camera
class Camera:
    def __init__(self, queue, scale: float):
        self.q = queue
        self.scale = scale

    def frame(self) -> np.ndarray:
        img = self.q.get().getCvFrame()
        if self.scale != 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return img


def enroll(cam: Camera, path: str) -> np.ndarray:
    """Live preview; C freezes a frame, drag a box around the target, ENTER saves it."""
    log("ENROLL: aim at the target, press C to capture, Q to quit")
    box_title = "Drag a box around the target. ENTER = save, C = cancel"
    while True:
        frame = cam.frame()
        view = frame.copy()
        put_text(view, "Aim at the target - C = capture, Q = quit", (10, 35), 0.9)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            sys.exit("Quit.")
        if key == ord("c"):
            x, y, w, h = map(int, cv2.selectROI(box_title, frame, showCrosshair=False))
            cv2.destroyWindow(box_title)
            if w < 30 or h < 30:
                log("Box too small or cancelled - try again")
                continue
            crop = frame[y:y + h, x:x + w].copy()
            cv2.imwrite(path, crop)
            log(f"Target saved to {path} ({w}x{h} px)")
            return crop


# --------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Find a specific target with the OAK-D Lite.")
    p.add_argument("--enroll", action="store_true", help="teach it the target from the live camera")
    p.add_argument("--target", default="target.png", help="target image to save to / load from")
    p.add_argument("--detector", choices=["sift", "orb"], default="sift",
                   help="sift = more robust (default), orb = faster")
    p.add_argument("--min-inliers", type=int, default=None,
                   help="matches needed to call it a detection (higher = fewer false alarms; "
                        "default 20 for sift, 15 for orb)")
    p.add_argument("--scale", type=float, default=0.75, help="shrink frames before matching (speed)")
    p.add_argument("--no-display", action="store_true", help="no window; print status only")
    return p.parse_args(argv)


def main():
    if int(dai.__version__.split(".")[0]) < 3:
        sys.exit("This script needs DepthAI v3. Run: python3 -m pip install -U depthai")

    args = parse_args()
    show = (not args.no_display) and bool(os.environ.get("DISPLAY"))
    if args.enroll and not show:
        sys.exit("--enroll needs a screen (it shows a live window). Use --target <photo> instead.")

    ref = None
    if not args.enroll:
        ref = cv2.imread(args.target)
        if ref is None:
            sys.exit(f"Can't read the target image '{args.target}'. "
                     "Run with --enroll first, or pass --target <photo>.")

    with dai.Pipeline() as pipeline:
        cam_node = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        out = cam_node.requestOutput((1280, 720), type=dai.ImgFrame.Type.BGR888p, fps=15)
        queue = out.createOutputQueue(maxSize=1, blocking=False)
        pipeline.start()
        cam = Camera(queue, args.scale)

        if args.enroll:
            ref = enroll(cam, args.target)

        try:
            min_inliers = args.min_inliers or (20 if args.detector == "sift" else 15)
            matcher = TargetMatcher(ref, args.detector, min_inliers)
        except ValueError as e:
            sys.exit(str(e))
        log(f"Target loaded: {matcher.n_ref} details ({matcher.kind.upper()}). Searching...")

        tracker = FoundTracker()
        last_status = last_report = time.monotonic()
        fps, t_prev = 0.0, time.monotonic()
        try:
            while pipeline.isRunning():
                frame = cam.frame()
                det = matcher.find(frame)
                event = tracker.update(det is not None)
                now = time.monotonic()
                fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
                t_prev = now

                if event == "FOUND":
                    log(f"*** TARGET FOUND ***  {where(det.center[0], frame.shape[1])}  "
                        f"(match strength {det.inliers})")
                    last_report = now
                elif event == "LOST":
                    log("target lost - searching...")
                    last_status = now
                elif tracker.found and det is not None and now - last_report > REPORT_EVERY_S:
                    log(f"target in view: {where(det.center[0], frame.shape[1])}  "
                        f"(match strength {det.inliers})")
                    last_report = now
                elif not tracker.found and now - last_status > STATUS_EVERY_S:
                    log("still searching...")
                    last_status = now

                if show:
                    cv2.imshow(WINDOW, draw_view(frame, det, tracker, fps))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            log("Ctrl+C - exiting")
        finally:
            if show:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()