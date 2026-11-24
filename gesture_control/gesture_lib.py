"""
gesture_lib.py - shared code for gesture_setup.py and gesture_flight.py.

Nothing in this file talks to the drone. It holds:
  * the OAK-D Lite camera pipeline (person detector with depth on the camera's chip,
    a high-resolution color picture for the hands, and the raw depth picture)
  * the hand model (MediaPipe Hand Landmarker, runs on the Jetson's CPU)
  * turning a hand into numbers ("features") and matching them against your saved gestures
  * the debouncer that decides when a gesture really counts (hold-to-move, land, wake)
  * a small person tracker so the drone keeps following the same person (the operator)
  * saving and loading the gesture profile (a JSON file, numbers only, no pictures)

Keep this file in the same folder as gesture_setup.py and gesture_flight.py.
"""

import json
import math
import os
import shutil
import sys
import time
import urllib.request
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# ------------------------------------------------------------------ gestures
WAKE, LEFT, RIGHT, FORWARD, BACKWARD, LAND = "wake", "left", "right", "forward", "backward", "land"
GESTURES = [WAKE, LEFT, RIGHT, FORWARD, BACKWARD, LAND]   # order used by the setup phase
MOVES = [LEFT, RIGHT, FORWARD, BACKWARD]                   # hold-to-move gestures
NEUTRAL = "neutral"     # a hand that is clearly NOT a signal (taught during setup)
UNKNOWN = "unknown"     # a hand that matches nothing well enough
NONE = "none"           # no hand seen
CONFLICT = "conflict"   # two hands showing two different signals
PASSIVE = {NEUTRAL, UNKNOWN, NONE}   # labels that mean "no command"

SUGGESTIONS = {
    WAKE: "e.g. open palm facing the camera, fingers up, held still",
    LEFT: "e.g. point your index finger to the side",
    RIGHT: "e.g. point your index finger to the other side",
    FORWARD: "e.g. flat hand, palm toward the camera, fingers pointing down",
    BACKWARD: "e.g. fist with the thumb up",
    LAND: "e.g. fist, held still (make it very different from the others)",
}

PROFILE_VERSION = 1
FEATURE_VERSION = "wrist-palm-xyz-v1"
Z_WEIGHT = 0.5            # MediaPipe's depth guess per point is noisier than x/y, so it counts less
K_NEAREST = 3             # a hand is compared with its 3 closest samples of each gesture
THRESHOLD_SCALE = 2.0     # match distance allowed = this x the spread seen during setup
THRESHOLD_FLOOR = 0.20    # ...but never stricter than this (very consistent samples)
MARGIN_MOVE = 1.25        # best match must beat the runner-up by 25% (movement gestures)
MARGIN_CRITICAL = 1.5     # ...and by 50% for land and wake
MIN_SAMPLES = 5           # fewest samples per gesture the setup accepts

DEFAULT_TIMING = {
    "start_s": 0.4,          # hold a movement gesture this long before the drone moves
    "release_s": 0.3,        # stop moving this soon after the gesture is gone
    "land_hold_s": 1.5,      # hold LAND this long
    "wake_hold_s": 1.0,      # hold WAKE this long
    "recheck_hold_s": 0.6,   # hold time during the pre-flight recheck
    "fraction": 0.8,         # share of frames in the hold window that must agree
}
MIN_WINDOW_FRAMES = 3

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
MODEL_DIR = Path.home() / ".cache" / "gesture_control"
SHAVE_TRIES = [6, 5, 4]
VIEW_WIDTH = 960
FONT = cv2.FONT_HERSHEY_SIMPLEX

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
    (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def display_available(no_display_flag: bool) -> bool:
    return (not no_display_flag) and bool(os.environ.get("DISPLAY"))


# ------------------------------------------------------------------ people
@dataclass
class Person:
    box: Tuple[float, float, float, float]   # normalized xmin, ymin, xmax, ymax
    confidence: float
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

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])


def nearest_person(people: Sequence[Person]) -> Optional[Person]:
    """Closest person with a depth reading; if none has depth, the biggest box."""
    with_depth = [p for p in people if p.has_depth]
    if with_depth:
        return min(with_depth, key=lambda p: p.z_mm)
    return max(people, key=lambda p: p.area, default=None)


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class PersonTracker:
    """Gives each person a number that stays the same from frame to frame (box overlap).

    A person who is not seen for `max_age` seconds is forgotten; if they come back they
    get a new number. The flight script uses that: a forgotten operator must WAKE again.
    """

    def __init__(self, max_age: float = 1.0, iou_min: float = 0.25, max_depth_jump_mm: float = 1000):
        self.max_age = max_age
        self.iou_min = iou_min
        self.max_depth_jump_mm = max_depth_jump_mm
        self.tracks: Dict[int, Tuple[Person, float]] = {}   # id -> (last person, last seen)
        self._next_id = 1

    def update(self, people: Sequence[Person], now: float) -> Dict[int, Person]:
        pairs = []
        for tid, (tp, _) in self.tracks.items():
            for i, p in enumerate(people):
                if tp.has_depth and p.has_depth and abs(tp.z_mm - p.z_mm) > self.max_depth_jump_mm:
                    continue
                v = iou(tp.box, p.box)
                if v >= self.iou_min:
                    pairs.append((v, tid, i))
        pairs.sort(reverse=True)
        seen: Dict[int, Person] = {}
        used_p = set()
        for _, tid, i in pairs:
            if tid in seen or i in used_p:
                continue
            seen[tid] = people[i]
            used_p.add(i)
        for i, p in enumerate(people):
            if i not in used_p:
                seen[self._next_id] = p
                self._next_id += 1
        for tid, p in seen.items():
            self.tracks[tid] = (p, now)
        for tid in [t for t, (_, last) in self.tracks.items() if now - last > self.max_age]:
            del self.tracks[tid]
        return seen

    def alive(self, tid: Optional[int]) -> bool:
        return tid is not None and tid in self.tracks

    def reset(self) -> None:
        self.tracks.clear()


# ------------------------------------------------------------------ camera
@dataclass
class Frame:
    t: float                       # time.monotonic() when it arrived
    bgr: np.ndarray                # high-resolution color picture (same crop as the AI's input)
    people: List[Person]
    depth_mm: Optional[np.ndarray]  # uint16, millimetres, 0 = no data (may be None)


class ShaveError(RuntimeError):
    """The camera has too few free processing cores (SHAVEs) for the AI model."""


class CameraRig:
    """OAK-D Lite: person detection with distance (on the camera) + color + depth pictures."""

    def __init__(self, model: str = "yolov6-nano", fps: float = 15, frame_width: int = 1280,
                 conf: float = 0.5, shaves: Optional[int] = None):
        self.model, self.fps, self.frame_width = model, fps, frame_width
        self.conf, self.shaves = conf, shaves
        self._stack: Optional[ExitStack] = None
        self.pipeline = None
        self.labels: List[str] = []
        self._last_host = None
        self._last_depth = None

    # -- opening
    def open(self) -> "CameraRig":
        import depthai as dai
        if int(dai.__version__.split(".")[0]) < 3:
            sys.exit("This needs DepthAI v3. Run: python3 -m pip install -U depthai")
        tries = [self.shaves] if self.shaves else SHAVE_TRIES
        for i, n in enumerate(tries):
            try:
                self._open_once(n)
                log(f"Camera running ({n} AI cores, picture {self.frame_size[0]}x{self.frame_size[1]})")
                return self
            except ShaveError as e:
                self.close()
                if i == len(tries) - 1:
                    sys.exit(f"The camera can't fit the AI model even with {n} cores: {e}")
                log(f"Not enough free camera cores at {n} - retrying with {tries[i + 1]}...")
        return self

    def _open_once(self, shaves: int) -> None:
        import depthai as dai
        self._stack = ExitStack()
        pipeline = self._stack.enter_context(dai.Pipeline())
        platform = pipeline.getDefaultDevice().getPlatformAsString()
        archive = dai.NNArchive(dai.getModelFromZoo(dai.NNModelDescription(self.model, platform=platform)))
        try:
            nn_w, nn_h = archive.getInputWidth(), archive.getInputHeight()
        except Exception:
            nn_w, nn_h = 4, 3

        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left.requestOutput((640, 400), fps=self.fps),
            right.requestOutput((640, 400), fps=self.fps),
            presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS,
        )
        nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(cam, stereo, archive, fps=self.fps)
        nn.setNNArchive(archive, numShaves=shaves)
        nn.input.setBlocking(False)
        nn.setConfidenceThreshold(self.conf)
        nn.setBoundingBoxScaleFactor(0.5)    # measure depth on the middle of the box
        nn.setDepthLowerThreshold(100)       # ignore depth closer than 0.1 m
        nn.setDepthUpperThreshold(10000)     # ignore depth beyond 10 m
        self.labels = list(nn.getClasses() or ["person"])

        # The AI input is a centre crop of the sensor (CROP mode). Asking for the big picture
        # with the same shape and the same CROP mode makes the person boxes line up with it.
        w = int(self.frame_width) // 2 * 2
        h = int(round(w * nn_h / nn_w / 2)) * 2
        host = cam.requestOutput((w, h), type=dai.ImgFrame.Type.NV12,
                                 resizeMode=dai.ImgResizeMode.CROP, fps=self.fps)
        self.frame_size = (w, h)
        self.q_det = nn.out.createOutputQueue(maxSize=1, blocking=False)
        self.q_host = host.createOutputQueue(maxSize=1, blocking=False)
        self.q_depth = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
        try:
            pipeline.start()
        except RuntimeError as e:
            if "shave" in str(e).lower():
                raise ShaveError(str(e).splitlines()[0])
            raise
        self.pipeline = pipeline

    # -- reading
    def is_running(self) -> bool:
        return self.pipeline is not None and self.pipeline.isRunning()

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        """Next detection result paired with the newest color and depth pictures."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if not self.is_running():
                return None
            msg = self.q_host.tryGet()
            if msg is not None:
                self._last_host = msg.getCvFrame()
            dmsg = self.q_depth.tryGet()
            if dmsg is not None:
                self._last_depth = dmsg.getFrame()
            det = self.q_det.tryGet()
            if det is not None and self._last_host is not None:
                return Frame(time.monotonic(), self._last_host, self._people(det.detections),
                             self._last_depth)
            time.sleep(0.003)
        return None

    def _people(self, detections) -> List[Person]:
        out = []
        for d in detections:
            if 0 <= d.label < len(self.labels):
                name = self.labels[d.label]
            else:
                name = d.labelName or str(d.label)
            if name != "person":
                continue
            sc = d.spatialCoordinates
            out.append(Person((float(d.xmin), float(d.ymin), float(d.xmax), float(d.ymax)),
                              float(d.confidence), float(sc.x), float(sc.y), float(sc.z)))
        return out

    def close(self) -> None:
        if self._stack is not None:
            try:
                self._stack.close()
            except Exception:
                pass
        self._stack = None
        self.pipeline = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ depth (forward obstacle)
ROI_WIDTH = 0.40
ROI_HEIGHT = 0.40
MIN_VALID_FRACTION = 0.10
NEAR_PERCENTILE = 5


def analyze_depth(depth_mm: Optional[np.ndarray], stop_dist_m: float) -> Tuple[Optional[float], bool, str]:
    """Same rule as obstacle_stop.py: (nearest_m, blocked, reason) for the centre of the view.
    No data counts as blocked."""
    if depth_mm is None:
        return None, True, "NO DEPTH PICTURE"
    h, w = depth_mm.shape[:2]
    x0, x1 = int(w * (0.5 - ROI_WIDTH / 2)), int(w * (0.5 + ROI_WIDTH / 2))
    y0, y1 = int(h * (0.5 - ROI_HEIGHT / 2)), int(h * (0.5 + ROI_HEIGHT / 2))
    roi = depth_mm[y0:y1, x0:x1]
    valid = roi[roi > 0]
    if valid.size < MIN_VALID_FRACTION * roi.size:
        return None, True, "NO DEPTH DATA"
    nearest = float(np.percentile(valid, NEAR_PERCENTILE)) / 1000.0
    if nearest < stop_dist_m:
        return nearest, True, "OBSTACLE"
    return nearest, False, "CLEAR"


# ------------------------------------------------------------------ hands
def ensure_hand_model(model_dir: Path = MODEL_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    p = model_dir / "hand_landmarker.task"
    if p.exists() and p.stat().st_size > 100_000:
        return p
    log("Downloading the hand model (one time only, ~8 MB)...")
    tmp = model_dir / "hand_landmarker.task.part"
    try:
        req = urllib.request.Request(HAND_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        os.replace(tmp, p)
    except Exception as e:
        sys.exit(f"Couldn't download the hand model: {e}\nDownload it by hand from:\n  {HAND_MODEL_URL}\n"
                 f"and save it as:\n  {p}")
    return p


class HandDetector:
    """MediaPipe Hand Landmarker: 21 points per hand, up to `num_hands` hands per picture."""

    def __init__(self, num_hands: int = 2, min_confidence: float = 0.5):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError:
            sys.exit("MediaPipe is missing. Run: python3 -m pip install -U mediapipe")
        self._mp = mp
        opts = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(ensure_hand_model())),
            running_mode=vision.RunningMode.IMAGE,   # each crop is independent
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
        )
        self._lm = vision.HandLandmarker.create_from_options(opts)

    def detect(self, bgr: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """Hands in `bgr` as (21x3 points in bgr's pixel coordinates, handedness, score).
        z is MediaPipe's relative depth, scaled like x."""
        h, w = bgr.shape[:2]
        side = max(h, w)
        # pad to a square so MediaPipe's coordinates are unambiguous
        sq = np.zeros((side, side, 3), np.uint8)
        sq[:h, :w] = bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))
        res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb))
        out = []
        for i, pts in enumerate(res.hand_landmarks or []):
            arr = np.array([[p.x * side, p.y * side, p.z * side] for p in pts], np.float32)
            hand, score = "?", 0.0
            if res.handedness and i < len(res.handedness) and res.handedness[i]:
                hand, score = res.handedness[i][0].category_name, float(res.handedness[i][0].score)
            out.append((arr, hand, score))
        return out

    def close(self) -> None:
        try:
            self._lm.close()
        except Exception:
            pass


def hand_features(lm: np.ndarray) -> Optional[np.ndarray]:
    """Turn 21 hand points into 60 numbers that don't depend on where the hand is or how big
    it looks: every point relative to the wrist, divided by the wrist-to-middle-knuckle length.
    The hand's direction (pointing left vs right) is kept on purpose."""
    rel = lm.astype(np.float32) - lm[0].astype(np.float32)
    scale = float(np.linalg.norm(rel[9, :2]))
    if scale < 1e-6:
        return None
    rel = rel / scale
    return np.concatenate([rel[1:, :2].ravel(), Z_WEIGHT * rel[1:, 2]]).astype(np.float32)


def palm_size_px(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[9, :2] - lm[0, :2]))


@dataclass
class Hand:
    lm: np.ndarray                 # 21x3, pixel coordinates in the FULL frame
    handedness: str
    palm_px: float
    feature: Optional[np.ndarray]
    label: str = UNKNOWN
    dist: float = float("inf")
    margin: float = 0.0

    @property
    def wrist_y(self) -> float:
        return float(self.lm[0, 1])


@dataclass
class HandsResult:
    label: str                         # combined label for this person this frame
    hands: List[Hand] = field(default_factory=list)
    crop: Optional[Tuple[int, int, int, int]] = None   # x0, y0, x1, y1 in frame pixels
    too_small: int = 0                 # hands ignored because they looked too small


def person_crop(person: Person, frame_shape, expand_x: float = 0.30, extend_top: float = 0.15,
                keep_height: float = 0.70) -> Optional[Tuple[int, int, int, int]]:
    """Upper-body crop around a person box (hands are tiny in the full picture at 3 m)."""
    H, W = frame_shape[:2]
    x0, y0, x1, y1 = person.box
    bw, bh = x1 - x0, y1 - y0
    cx0 = int(max(0.0, x0 - expand_x * bw) * W)
    cx1 = int(min(1.0, x1 + expand_x * bw) * W)
    cy0 = int(max(0.0, y0 - extend_top * bh) * H)
    cy1 = int(min(1.0, y0 + keep_height * bh) * H)
    if cx1 - cx0 < 32 or cy1 - cy0 < 32:
        return None
    return cx0, cy0, cx1, cy1


def combine_labels(labels: Sequence[str]) -> str:
    """One label for a person from the labels of each of their hands."""
    if not labels:
        return NONE
    signals = {l for l in labels if l in GESTURES}
    if len(signals) == 1:
        return signals.pop()
    if len(signals) > 1:
        return CONFLICT
    return NEUTRAL if NEUTRAL in labels else UNKNOWN


def analyze_hands(frame: Frame, person: Person, detector: HandDetector,
                  classifier: Optional["GestureClassifier"], min_palm_px: float = 10.0) -> HandsResult:
    crop = person_crop(person, frame.bgr.shape)
    if crop is None:
        return HandsResult(NONE)
    x0, y0, x1, y1 = crop
    hands, small = [], 0
    for lm, hand, _ in detector.detect(frame.bgr[y0:y1, x0:x1]):
        lm = lm.copy()
        lm[:, 0] += x0
        lm[:, 1] += y0
        palm = palm_size_px(lm)
        if palm < min_palm_px:
            small += 1
            continue
        h = Hand(lm, hand, palm, hand_features(lm))
        if classifier is not None and h.feature is not None:
            h.label, h.dist, h.margin = classifier.classify(h.feature)
        hands.append(h)
    label = combine_labels([h.label for h in hands]) if classifier is not None else (
        NONE if not hands else UNKNOWN)
    return HandsResult(label, hands, crop, small)


# ------------------------------------------------------------------ classifier
class GestureClassifier:
    """Nearest-neighbour matching against the samples saved in the profile."""

    def __init__(self, profile: dict):
        self.samples: Dict[str, np.ndarray] = {}
        self.threshold: Dict[str, float] = {}
        self.margin: Dict[str, float] = {}
        for name, c in profile["classes"].items():
            s = np.asarray(c.get("samples", []), np.float32)
            if s.size == 0:
                continue
            self.samples[name] = s.reshape(len(s), -1)
            self.threshold[name] = float(c.get("threshold", float("inf")))
            self.margin[name] = float(c.get("margin", MARGIN_MOVE))

    def class_distance(self, f: np.ndarray, name: str, exclude: Optional[int] = None) -> float:
        s = self.samples[name]
        if exclude is not None:
            s = np.delete(s, exclude, axis=0)
        if len(s) == 0:
            return float("inf")
        d = np.linalg.norm(s - f, axis=1)
        k = min(K_NEAREST, len(d))
        return float(np.mean(np.partition(d, k - 1)[:k]))

    def classify(self, f: np.ndarray, exclude: Optional[Tuple[str, int]] = None) -> Tuple[str, float, float]:
        """(label, distance to the best gesture, how much better it is than the runner-up)."""
        dists = []
        for name in self.samples:
            ex = exclude[1] if exclude is not None and exclude[0] == name else None
            dists.append((self.class_distance(f, name, ex), name))
        dists = [d for d in dists if math.isfinite(d[0])]
        if not dists:
            return UNKNOWN, float("inf"), 0.0
        dists.sort()
        d1, best = dists[0]
        d2 = dists[1][0] if len(dists) > 1 else float("inf")
        ratio = d2 / max(d1, 1e-6)
        if best == NEUTRAL:
            return NEUTRAL, d1, ratio
        if d1 > self.threshold[best] or ratio < self.margin[best]:
            return UNKNOWN, d1, ratio
        return best, d1, ratio


# ------------------------------------------------------------------ profile
def build_profile(samples: Dict[str, List[np.ndarray]], name: str, timing: Optional[dict] = None,
                  meta: Optional[dict] = None) -> dict:
    """Make a profile from captured samples and work out each gesture's match distance."""
    classes = {}
    for label, feats in samples.items():
        arr = np.asarray(feats, np.float32).reshape(len(feats), -1) if len(feats) else np.zeros((0, 60))
        entry = {"samples": np.round(arr, 4).tolist()}
        if label != NEUTRAL:
            spread = []
            for i in range(len(arr)):
                rest = np.delete(arr, i, axis=0)
                if len(rest):
                    d = np.linalg.norm(rest - arr[i], axis=1)
                    k = min(K_NEAREST, len(d))
                    spread.append(float(np.mean(np.partition(d, k - 1)[:k])))
            p90 = float(np.percentile(spread, 90)) if spread else THRESHOLD_FLOOR
            entry["threshold"] = round(max(THRESHOLD_FLOOR, THRESHOLD_SCALE * p90), 4)
            entry["margin"] = MARGIN_CRITICAL if label in (LAND, WAKE) else MARGIN_MOVE
        classes[label] = entry
    return {
        "version": PROFILE_VERSION,
        "feature": FEATURE_VERSION,
        "name": name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "timing": dict(DEFAULT_TIMING, **(timing or {})),
        "meta": meta or {},
        "report": {},
        "tests": {},
        "validated": False,
    }


def evaluate_profile(profile: dict) -> dict:
    """Leave-one-out check: classify every saved sample against all the others.

    Returns {"accuracy": {...}, "confusion": {...}, "closest_pair": ..., "critical": [...], "warnings": [...]}.
    """
    clf = GestureClassifier(profile)
    confusion: Dict[str, Dict[str, int]] = {}
    accuracy: Dict[str, float] = {}
    critical, warnings = [], []
    for g in GESTURES:
        n = len(profile["classes"].get(g, {}).get("samples", []))
        if n < MIN_SAMPLES:
            critical.append(f"{g.upper()}: only {n} samples (need {MIN_SAMPLES}+)")
    for label, s in clf.samples.items():
        row: Dict[str, int] = {}
        for i in range(len(s)):
            pred, _, _ = clf.classify(s[i], exclude=(label, i))
            row[pred] = row.get(pred, 0) + 1
        confusion[label] = row
        if label != NEUTRAL:
            accuracy[label] = row.get(label, 0) / len(s)
    for label, row in confusion.items():
        for crit in (LAND, WAKE):
            if label != crit and row.get(crit, 0):
                critical.append(f"{row[crit]} {label.upper()} sample(s) were read as {crit.upper()}")
        if label == NEUTRAL:
            moved = sum(row.get(m, 0) for m in MOVES)
            if moved:
                warnings.append(f"{moved} relaxed-hand (neutral) sample(s) were read as a movement signal")
            continue
        wrong = {k: v for k, v in row.items() if k in GESTURES and k != label}
        if wrong:
            warnings.append(f"{label.upper()} sometimes read as " +
                            ", ".join(f"{k.upper()} x{v}" for k, v in wrong.items()))
    for label, acc in accuracy.items():
        if acc < 0.7:
            critical.append(f"{label.upper()} recognized in only {acc:.0%} of its own samples")
        elif acc < 0.9:
            warnings.append(f"{label.upper()} recognized in {acc:.0%} of its own samples")
    n_neutral = len(profile["classes"].get(NEUTRAL, {}).get("samples", []))
    if n_neutral < 10:
        warnings.append(f"only {n_neutral} relaxed-hand samples (10+ helps prevent accidental commands)")

    # how close is the most similar pair of signals?
    closest = None
    present = [g for g in GESTURES if g in clf.samples]
    for i, a in enumerate(present):
        for b in present[i + 1:]:
            cross = np.mean([clf.class_distance(f, b) for f in clf.samples[a]])
            spread = max(clf.threshold[a], clf.threshold[b]) / THRESHOLD_SCALE
            ratio = float(cross / max(spread, 1e-6))
            if closest is None or ratio < closest[2]:
                closest = (a, b, ratio)
    if closest and closest[2] < 2.0:
        warnings.append(f"{closest[0].upper()} and {closest[1].upper()} look alike "
                        f"(separation {closest[2]:.1f}; 3+ is comfortable)")
    return {"accuracy": accuracy, "confusion": confusion,
            "closest_pair": list(closest) if closest else None,
            "critical": critical, "warnings": warnings}


def save_profile(path: str, profile: dict) -> None:
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(profile, f, indent=1)
    os.replace(tmp, path)


def load_profile(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        prof = json.load(f)
    if prof.get("version") != PROFILE_VERSION or prof.get("feature") != FEATURE_VERSION:
        raise ValueError(f"{path} was made by a different version of the setup. Run gesture_setup.py again.")
    missing = [g for g in GESTURES if not prof.get("classes", {}).get(g, {}).get("samples")]
    if missing:
        raise ValueError(f"{path} has no samples for: {', '.join(missing)}")
    prof["timing"] = dict(DEFAULT_TIMING, **prof.get("timing", {}))
    return prof


# ------------------------------------------------------------------ debouncer
class GestureDebouncer:
    """Decides when a gesture really counts, from the stream of per-frame labels.

    Movement (hold to move): a movement gesture becomes `active` after it was held for
    start_s, and stops being active `release_s` after it disappears, or at once when a
    different signal (or two conflicting signals) shows up.
    LAND / WAKE: reported once as an event after being held for land_hold_s / wake_hold_s;
    the gesture must go away before it can be reported again.
    """

    def __init__(self, timing: dict):
        self.t = dict(DEFAULT_TIMING, **timing)
        self.horizon = max(self.t["land_hold_s"], self.t["wake_hold_s"], self.t["start_s"],
                           self.t["recheck_hold_s"]) + 1.0
        self.reset()

    def reset(self) -> None:
        self.hist: deque = deque()
        self.active: Optional[str] = None
        self.active_since = 0.0
        self._last_active_seen = 0.0
        self._fired = set()

    @property
    def last_label(self) -> str:
        return self.hist[-1][1] if self.hist else NONE

    def held(self, label: str, hold_s: float) -> bool:
        """Was `label` shown for (nearly) all of the last hold_s seconds, including now?"""
        if not self.hist or self.hist[-1][1] != label:
            return False
        now = self.hist[-1][0]
        start = now - hold_s
        if self.hist[0][0] > start + 0.1 * hold_s:      # not watching for long enough yet
            return False
        window = [l for (t, l) in self.hist if t >= start]
        if len(window) < MIN_WINDOW_FRAMES:
            return False
        return sum(1 for l in window if l == label) / len(window) >= self.t["fraction"]

    def update(self, now: float, label: str) -> List[str]:
        self.hist.append((now, label))
        while self.hist and now - self.hist[0][0] > self.horizon:
            self.hist.popleft()

        if self.active is not None:
            if label == self.active:
                self._last_active_seen = now
            elif label not in PASSIVE:
                self.active = None          # a different signal: stop at once
            if self.active is not None and now - self._last_active_seen > self.t["release_s"]:
                self.active = None
        if self.active is None:
            for g in MOVES:
                if self.held(g, self.t["start_s"]):
                    self.active, self.active_since, self._last_active_seen = g, now, now
                    break

        events = []
        for g, hold in ((LAND, self.t["land_hold_s"]), (WAKE, self.t["wake_hold_s"])):
            if self.held(g, hold):
                if g not in self._fired:
                    self._fired.add(g)
                    events.append(g)
            elif label != g:
                self._fired.discard(g)
        return events


# ------------------------------------------------------------------ drawing
LABEL_COLORS = {
    WAKE: (255, 200, 0), LAND: (0, 0, 255), LEFT: (0, 220, 0), RIGHT: (0, 220, 0),
    FORWARD: (0, 220, 0), BACKWARD: (0, 220, 0), NEUTRAL: (180, 180, 180),
    UNKNOWN: (0, 165, 255), CONFLICT: (0, 0, 200), NONE: (120, 120, 120),
}


def put_text(img, text, org, scale=0.7, thickness=2, color=(255, 255, 255)):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, org, FONT, scale, color, thickness)


def draw_view(frame_bgr: np.ndarray, tracked: Dict[int, Person], highlight_id: Optional[int],
              results: Dict[int, HandsResult], top_lines: Sequence[str], banner_color=(60, 60, 60),
              bottom_lines: Sequence[str] = ()) -> np.ndarray:
    s = VIEW_WIDTH / frame_bgr.shape[1]
    view = cv2.resize(frame_bgr, None, fx=s, fy=s)
    h, w = view.shape[:2]
    for tid, p in tracked.items():
        x0, y0, x1, y1 = int(p.box[0] * w), int(p.box[1] * h), int(p.box[2] * w), int(p.box[3] * h)
        mine = tid == highlight_id
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0) if mine else (150, 150, 150), 3 if mine else 1)
        d = p.distance_m
        put_text(view, f"#{tid}" + (f" {d:.1f} m" if d else ""), (x0 + 4, min(h - 8, y1 - 8)), 0.55, 1)
    for tid, r in results.items():
        if r.crop:
            cx0, cy0, cx1, cy1 = [int(v * s) for v in r.crop]
            cv2.rectangle(view, (cx0, cy0), (cx1, cy1), (255, 255, 0), 1)
        for hand in r.hands:
            pts = (hand.lm[:, :2] * s).astype(int)
            col = LABEL_COLORS.get(hand.label, (255, 255, 255))
            for a, b in HAND_CONNECTIONS:
                cv2.line(view, tuple(pts[a]), tuple(pts[b]), col, 2)
            put_text(view, f"{hand.label} {hand.dist:.2f}" if math.isfinite(hand.dist) else hand.label,
                     (int(pts[0][0]) + 6, int(pts[0][1]) + 18), 0.55, 1, col)
    bh = 30 * max(1, len(top_lines)) + 10
    cv2.rectangle(view, (0, 0), (w, bh), banner_color, -1)
    for i, line in enumerate(top_lines):
        put_text(view, line, (10, 30 + 30 * i), 0.75 if i == 0 else 0.6, 2 if i == 0 else 1)
    for i, line in enumerate(reversed(list(bottom_lines))):
        put_text(view, line, (10, h - 12 - 26 * i), 0.6, 1)
    return view
