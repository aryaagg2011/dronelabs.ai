#!/usr/bin/env python3
"""
Find people with the OAK-D Lite: ANY person, or ONE SPECIFIC person you enroll.

The person detector (YOLOv6-Nano) runs ON THE CAMERA'S OWN CHIP, and the camera
measures how far away each person is. To pick out one specific person, the
Jetson also runs face recognition (OpenCV YuNet + SFace) and compares each
face it sees with photos of the person you enrolled.

USAGE
  Any person:
    python3 find_person.py
    python3 find_person.py --label bottle          # other object types
    python3 find_person.py --list-labels

  One specific person:
    python3 find_person.py --enroll --name Alex    # 1) teach it (live camera), then it starts searching
    python3 find_person.py --specific              # 2) search for the enrolled person any time later

    Enrolling from existing photos instead (no screen needed):
    python3 find_person.py --enroll-photo a.jpg b.jpg c.jpg --name Alex

  ENROLL WINDOW: the person looks at the camera. Press SPACE to capture a
  sample (exactly one face must be in view). Take 5-8 samples, turning the head
  a little and changing distance and lighting between them, then press ENTER.

WHAT IT PRINTS
  *** ALEX FOUND ***  3.1 m away, 8 deg right (face match 0.62) | 2 people in view
  ALEX in view: 3.0 m away, 9 deg right (face match 0.58)
  alex lost - searching...

HONEST LIMITS (specific-person mode)
  * It recognizes FACES. The face must be visible, facing roughly toward the
    camera, and large enough in the picture - about 2-3 m or closer. It cannot
    recognize someone from behind or from far away.
  * Only enroll people who have agreed to it. The enrolled data (numbers
    describing the face, not photos) is stored in enrolled_person.npz on this
    computer and never leaves it.
  * Distance/direction come from the camera's stereo depth (good to ~5-6 m).

NOTES
  * First run downloads the person model from Luxonis, and (for specific mode)
    two small face models from the OpenCV model zoo. Internet needed once.
  * Needs a screen for the live window; use --no-display over SSH.

Requires DepthAI v3:   python3 -m pip install -U depthai opencv-python numpy
"""

import argparse
import difflib
import math
import os
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
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
SPECIFIC_LOST_AFTER_S = 2.0   # face matches flicker more, so wait longer before LOST
REPORT_EVERY_S = 1.0      # position update interval while target is in view
STATUS_EVERY_S = 5.0      # "still searching" message interval
SHAVE_TRIES = [6, 5, 4]   # camera cores to give the AI model, tried in this order
DISPLAY_SCALE = 2.0       # the AI's input picture is small; enlarge it for the window
VIEW_WIDTH = 960          # window width in specific-person mode
FACE_THRESHOLD = 0.40     # face similarity needed for a match (higher = stricter)
MIN_ENROLL_SAMPLES = 3
MAX_ENROLL_SAMPLES = 10
MODEL_DIR = Path.home() / ".cache" / "find_person"
FACE_MODELS = [
    ("face_detection_yunet_2023mar.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"),
    ("face_recognition_sface_2021dec.onnx",
     "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"),
]
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

    def position_text(self) -> str:
        d = self.distance_m
        dist = f"{d:.1f} m away" if d is not None else "distance unknown"
        return f"{dist}, {self.where()}"

    def describe(self) -> str:
        return f"{self.name} {self.position_text()} (confidence {self.confidence:.0%})"


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

    def __init__(self, lost_after: float = LOST_AFTER_S):
        self.found = False
        self.hits = 0
        self.last_seen = 0.0
        self.lost_after = lost_after

    def update(self, detected: bool, now: float) -> Optional[str]:
        if detected:
            self.hits += 1
            self.last_seen = now
        else:
            self.hits = 0
        if not self.found and self.hits >= FOUND_NEEDED:
            self.found = True
            return "FOUND"
        if self.found and now - self.last_seen > self.lost_after:
            self.found = False
            return "LOST"
        return None


# --------------------------------------------------------- face recognition
def ensure_face_models(model_dir: Path = MODEL_DIR) -> Tuple[Path, Path]:
    """Return paths to the two face models, downloading them the first time."""
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fname, url in FACE_MODELS:
        p = model_dir / fname
        if not p.exists() or p.stat().st_size < 1000:
            log(f"Downloading face model {fname} (one time only)...")
            tmp = model_dir / (fname + ".part")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                os.replace(tmp, p)
            except Exception as e:
                sys.exit(f"Couldn't download {fname}: {e}\n"
                         f"Download it by hand from:\n  {url}\nand save it as:\n  {p}")
        paths.append(p)
    return paths[0], paths[1]


class FaceTools:
    def __init__(self, model_dir: Path = MODEL_DIR):
        if not (hasattr(cv2, "FaceDetectorYN") and hasattr(cv2, "FaceRecognizerSF")):
            sys.exit("Your OpenCV is too old for face recognition. "
                     "Run: python3 -m pip install -U opencv-python")
        yunet, sface = ensure_face_models(model_dir)
        self.detector = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.85, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface), "")

    def detect(self, bgr: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Faces as an N x 15 array in the coordinates of `bgr`:
        x, y, w, h, five landmark points (10 numbers), score."""
        small = bgr if scale == 1.0 else cv2.resize(bgr, None, fx=scale, fy=scale,
                                                    interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(small)
        if faces is None:
            return np.zeros((0, 15), np.float32)
        faces = faces.copy()
        if scale != 1.0:
            faces[:, :14] /= scale
        return faces

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        e = self.recognizer.feature(self.recognizer.alignCrop(bgr, face_row)).ravel().astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-9)


@dataclass
class Enrolled:
    name: str
    embeddings: np.ndarray   # N x 128, each row unit length


def save_enrolled(path: str, name: str, embeddings: List[np.ndarray]) -> None:
    np.savez(path, name=name, embeddings=np.array(embeddings, dtype=np.float32))


def load_enrolled(path: str) -> Optional[Enrolled]:
    if not os.path.exists(path):
        return None
    data = np.load(path, allow_pickle=False)
    return Enrolled(str(data["name"]), data["embeddings"])


def best_similarity(emb: np.ndarray, enrolled: Enrolled) -> float:
    return float(np.max(enrolled.embeddings @ emb))


@dataclass
class FaceResult:
    box: Tuple[float, float, float, float]   # x, y, w, h in host-picture pixels
    sim: float                               # similarity to the enrolled person
    owner: Optional[Hit]                     # the person-detection box this face sits inside


@dataclass
class Match:
    name: str
    face: FaceResult
    hit: Hit                                 # the person (with distance) or a face-only stand-in


def find_enrolled_face(host: np.ndarray, people: List[Hit], tools: FaceTools,
                       enrolled: Enrolled, threshold: float, det_scale: float):
    """Look at every face in the picture. Returns (all faces, best match or None)."""
    H, W = host.shape[:2]
    results: List[FaceResult] = []
    for f in tools.detect(host, det_scale):
        sim = best_similarity(tools.embed(host, f), enrolled)
        cx, cy = (f[0] + f[2] / 2) / W, (f[1] + f[3] / 2) / H
        inside = [p for p in people if p.box[0] <= cx <= p.box[2] and p.box[1] <= cy <= p.box[3]]
        owner = min(inside, key=lambda p: (p.box[2] - p.box[0]) * (p.box[3] - p.box[1])) if inside else None
        results.append(FaceResult((float(f[0]), float(f[1]), float(f[2]), float(f[3])), sim, owner))

    best = max(results, key=lambda r: r.sim, default=None)
    if best is None or best.sim < threshold:
        return results, None
    hit = best.owner
    if hit is None:   # face seen but no person box around it: use the face position, no distance
        x, y, w, h = best.box
        hit = Hit("person", best.sim, (x / W, y / H, (x + w) / W, (y + h) / H), 0.0, 0.0, 0.0)
    return results, Match(enrolled.name, best, hit)


def enroll_from_photos(paths: List[str], name: str, out_path: str, tools: FaceTools) -> None:
    embs = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            log(f"{p}: can't read this file - skipped")
            continue
        scale = min(1.0, 1280 / max(img.shape[:2]))
        faces = tools.detect(img, scale)
        if len(faces) == 0:
            log(f"{p}: no face found - skipped")
            continue
        if len(faces) > 1:
            log(f"{p}: {len(faces)} faces found - using the largest")
        f = faces[np.argmax(faces[:, 2] * faces[:, 3])]
        embs.append(tools.embed(img, f))
        log(f"{p}: face added")
    if not embs:
        sys.exit("No usable face in any of the photos. Use clear, front-facing photos.")
    save_enrolled(out_path, name, embs)
    log(f"Saved '{name}' ({len(embs)} samples) to {out_path}")


def enroll_live(args, name: str, tools: FaceTools) -> None:
    """Live camera window: SPACE captures a sample, ENTER saves."""
    host_w = args.face_width
    host_h = host_w * 3 // 4 // 2 * 2
    det_scale = min(1.0, 960 / host_w)
    embs: List[np.ndarray] = []
    note = "Look at the camera. SPACE = capture, ENTER = finish, Q = quit"
    log(f"ENROLL '{name}': {note}")
    try:
        with dai.Pipeline() as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            out = cam.requestOutput((host_w, host_h), type=dai.ImgFrame.Type.NV12, fps=15)
            q = out.createOutputQueue(maxSize=1, blocking=False)
            pipeline.start()
            while pipeline.isRunning():
                frame = q.get().getCvFrame()
                faces = tools.detect(frame, det_scale)
                view_scale = VIEW_WIDTH / frame.shape[1]
                view = cv2.resize(frame, None, fx=view_scale, fy=view_scale)
                for f in faces:
                    x, y, w, h = [int(v * view_scale) for v in f[:4]]
                    cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0) if len(faces) == 1 else (0, 0, 255), 2)
                cv2.rectangle(view, (0, 0), (view.shape[1], 70), (60, 60, 60), -1)
                put_text(view, f"Enrolling '{name}': {len(embs)} samples", (10, 28), 0.8)
                put_text(view, note, (10, 58), 0.6, 1)
                cv2.imshow(WINDOW, view)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    sys.exit("Quit - nothing saved.")
                if key == ord(" "):
                    if len(faces) != 1:
                        note = f"{len(faces)} faces in view - need exactly ONE. Try again."
                    else:
                        embs.append(tools.embed(frame, faces[0]))
                        note = f"Captured {len(embs)}. Turn your head slightly / move, then SPACE again."
                    log(note)
                elif key in (10, 13):
                    if len(embs) >= MIN_ENROLL_SAMPLES:
                        break
                    note = f"Need at least {MIN_ENROLL_SAMPLES} samples (have {len(embs)})."
                    log(note)
                if len(embs) >= MAX_ENROLL_SAMPLES:
                    break
    except RuntimeError as e:
        sys.exit(f"Camera error while enrolling: {e}\nTry a smaller picture: --face-width 960")
    finally:
        cv2.destroyAllWindows()
    save_enrolled(args.person_file, name, embs)
    log(f"Saved '{name}' ({len(embs)} samples) to {args.person_file}")


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


def draw_specific_view(host, people: List[Hit], faces: List[FaceResult], match: Optional[Match],
                       tracker: FoundTracker, name: str, fps: float):
    scale = VIEW_WIDTH / host.shape[1]
    view = cv2.resize(host, None, fx=scale, fy=scale)
    h, w = view.shape[:2]
    for p in people:
        x0, y0, x1, y1 = int(p.box[0] * w), int(p.box[1] * h), int(p.box[2] * w), int(p.box[3] * h)
        is_target = match is not None and match.hit is p
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0) if is_target else (150, 150, 150),
                      3 if is_target else 1)
    for fr in faces:
        x, y, fw, fh = [int(v * scale) for v in fr.box]
        is_target = match is not None and match.face is fr
        cv2.rectangle(view, (x, y), (x + fw, y + fh), (0, 255, 0) if is_target else (0, 0, 255),
                      3 if is_target else 1)
        put_text(view, f"{name} {fr.sim:.2f}" if is_target else f"unknown {fr.sim:.2f}",
                 (x, max(18, y - 6)), 0.6, 1)

    cv2.rectangle(view, (0, 0), (w, 50), (0, 170, 0) if tracker.found else (60, 60, 60), -1)
    if tracker.found and match is not None:
        put_text(view, f"{name.upper()} FOUND  {match.hit.where()}", (10, 35), 0.9)
    else:
        put_text(view, f"searching for {name}  ({len(people)} people in view)", (10, 35), 0.8)
    put_text(view, f"{fps:.0f} fps", (w - 110, 35), 0.7)
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
    p.add_argument("--shaves", type=int, default=None,
                   help="processing cores the AI model uses on the camera (default: auto, tries 6 then 5, 4)")
    p.add_argument("--show-all", action="store_true",
                   help="also outline every other object the AI sees (gray)")
    p.add_argument("--no-display", action="store_true", help="no window; print status only")
    p.add_argument("--list-labels", action="store_true", help="list everything the model can detect")

    g = p.add_argument_group("specific person")
    g.add_argument("--enroll", action="store_true",
                   help="teach it one person using the live camera, then search for them")
    g.add_argument("--enroll-photo", nargs="+", metavar="PHOTO",
                   help="teach it one person from photo files, then search for them")
    g.add_argument("--specific", action="store_true",
                   help="search for the enrolled person only (ignores everyone else)")
    g.add_argument("--name", default="target", help="name for the person being enrolled")
    g.add_argument("--person-file", default="enrolled_person.npz",
                   help="where the enrolled person is saved/loaded")
    g.add_argument("--threshold", type=float, default=FACE_THRESHOLD,
                   help=f"face match strictness (default {FACE_THRESHOLD}; higher = fewer wrong matches)")
    g.add_argument("--face-width", type=int, default=1280,
                   help="width of the high-resolution picture used for faces (default 1280; try 960 if the camera complains)")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.list_labels:
        print("The model can detect these 80 types:\n")
        print(", ".join(COCO_CLASSES))
        return

    specific_mode = args.specific or args.enroll or bool(args.enroll_photo)

    if specific_mode:
        wanted = {"person"}
    else:
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

    tools = enrolled = None
    if specific_mode:
        if args.enroll and not show:
            sys.exit("--enroll needs a screen (it shows a live window). "
                     "Use --enroll-photo <photos> instead.")
        if not (args.enroll or args.enroll_photo):
            enrolled = load_enrolled(args.person_file)
            if enrolled is None:
                sys.exit(f"No enrolled person found ({args.person_file}). "
                         "Run with --enroll (live camera) or --enroll-photo <photos> first.")
        tools = FaceTools()
        if args.enroll:
            enroll_live(args, args.name, tools)
        elif args.enroll_photo:
            enroll_from_photos(args.enroll_photo, args.name, args.person_file, tools)
        enrolled = load_enrolled(args.person_file)
        banner_name = enrolled.name
        log(f"Searching for '{enrolled.name}' ({len(enrolled.embeddings)} enrolled samples, "
            f"match threshold {args.threshold})")

    log("Starting camera + AI model (the first run downloads the model - needs internet)...")
    tries = [args.shaves] if args.shaves else SHAVE_TRIES
    run_with_retries(
        lambda n: run_once(args, wanted, banner_name, show, n, tools, enrolled), tries)


class ShaveError(RuntimeError):
    """The camera has too few free processing cores (SHAVEs) for the AI model."""


def run_with_retries(run, tries):
    """Call run(shaves) with each shave count in turn until one fits on the camera."""
    for i, n in enumerate(tries):
        try:
            return run(n)
        except ShaveError as e:
            if i == len(tries) - 1:
                sys.exit(f"The camera can't fit the AI model even with {n} cores: {e}")
            log(f"Not enough free camera cores at {n} - retrying with {tries[i + 1]}...")


def run_once(args, wanted, banner_name, show, shaves, tools=None, enrolled=None):
    with dai.Pipeline() as pipeline:
        # Load the model and tell the camera how many of its processing cores it may use.
        # (The depth pipeline needs some of them too, so the model can't take all.)
        platform = pipeline.getDefaultDevice().getPlatformAsString()
        archive = dai.NNArchive(dai.getModelFromZoo(dai.NNModelDescription(args.model, platform=platform)))

        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        stereo = pipeline.create(dai.node.StereoDepth).build(
            left.requestOutput((640, 400), fps=args.fps),
            right.requestOutput((640, 400), fps=args.fps),
            presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS,
        )
        nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
            cam, stereo, archive, fps=args.fps
        )
        nn.setNNArchive(archive, numShaves=shaves)
        nn.input.setBlocking(False)
        nn.setConfidenceThreshold(args.conf)
        nn.setBoundingBoxScaleFactor(0.5)    # measure depth on the middle of the box, not the edges
        nn.setDepthLowerThreshold(100)       # ignore depth closer than 0.1 m
        nn.setDepthUpperThreshold(10000)     # ignore depth beyond 10 m
        labels = nn.getClasses() or COCO_CLASSES
        q_det = nn.out.createOutputQueue(maxSize=1, blocking=False)

        q_rgb = q_host = None
        if enrolled is None:
            q_rgb = nn.passthrough.createOutputQueue(maxSize=1, blocking=False)
        else:
            # A bigger picture, same 4:3 shape as the AI's input, so faces are large enough to
            # recognize. Person boxes (normalized 0-1) then line up with it directly.
            host_w = args.face_width
            host_h = host_w * 3 // 4 // 2 * 2
            host_out = cam.requestOutput((host_w, host_h), type=dai.ImgFrame.Type.NV12, fps=args.fps)
            q_host = host_out.createOutputQueue(maxSize=1, blocking=False)

        try:
            pipeline.start()
        except RuntimeError as e:
            if "shave" in str(e).lower():
                raise ShaveError(str(e).splitlines()[0])
            raise
        log(f"Running with {shaves} AI cores.")

        try:
            if enrolled is None:
                log(f"Looking for: {banner_name}   (confidence >= {args.conf:.0%})")
                loop_any(pipeline, q_det, q_rgb, labels, wanted, banner_name, show, args)
            else:
                log(f"Looking for: {enrolled.name}")
                loop_specific(pipeline, q_det, q_host, labels, tools, enrolled, show, args)
        except KeyboardInterrupt:
            log("Ctrl+C - exiting")
        finally:
            if show:
                cv2.destroyAllWindows()


def loop_any(pipeline, q_det, q_rgb, labels, wanted, banner_name, show, args):
    """Report any detection of the wanted object types."""
    tracker = FoundTracker()
    last_status = last_report = time.monotonic()
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
            cv2.imshow(WINDOW, draw_view(frame, targets, others, tracker, banner_name, args.show_all))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


def loop_specific(pipeline, q_det, q_host, labels, tools, enrolled, show, args):
    """Report only the enrolled person, recognized by face."""
    det_scale = min(1.0, 960 / args.face_width)
    tracker = FoundTracker(SPECIFIC_LOST_AFTER_S)
    name = enrolled.name
    host = None
    fps, t_prev = 0.0, time.monotonic()
    last_status = last_report = time.monotonic()
    while pipeline.isRunning():
        det_msg = q_det.get()
        msg = q_host.tryGet()
        if msg is not None:
            host = msg.getCvFrame()
        if host is None:
            continue
        people, _ = extract_hits(det_msg.detections, labels, {"person"})

        faces: List[FaceResult] = []
        match: Optional[Match] = None
        if people:   # only spend CPU on faces when the camera sees a person
            faces, match = find_enrolled_face(host, people, tools, enrolled, args.threshold, det_scale)

        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-3)
        t_prev = now
        event = tracker.update(match is not None, now)

        if event == "FOUND":
            log(f"*** {name.upper()} FOUND ***  {match.hit.position_text()} "
                f"(face match {match.face.sim:.2f}) | {len(people)} people in view")
            last_report = now
        elif event == "LOST":
            log(f"{name} lost - searching...")
            last_status = now
        elif tracker.found and match is not None and now - last_report > REPORT_EVERY_S:
            log(f"{name} in view: {match.hit.position_text()} (face match {match.face.sim:.2f})")
            last_report = now
        elif not tracker.found and now - last_status > STATUS_EVERY_S:
            seen = f"{len(people)} people in view, none matched" if people else "no people in view"
            log(f"still searching... ({seen})")
            last_status = now

        if show:
            cv2.imshow(WINDOW, draw_specific_view(host, people, faces, match, tracker, name, fps))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


if __name__ == "__main__":
    main()