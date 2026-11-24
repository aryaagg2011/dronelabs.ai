#!/usr/bin/env python3
"""
gesture_setup.py - teach the drone YOUR hand signals, check them, and save them.

Run this once (and again whenever recognition gets worse). Nothing here touches the drone.
The result is a profile file (gesture_profile.json) that gesture_flight.py loads every flight.

USAGE
    python3 gesture_setup.py --name Arya                  # full setup
    python3 gesture_setup.py --redo land wake             # re-teach only some gestures
    python3 gesture_setup.py --test-only                  # re-run the tests on a saved profile
  No monitor on the Jetson? Run it through the browser viewer:
    python3 remote_view.py gesture_setup.py --name Arya

WHAT HAPPENS
  1. Teach each signal: WAKE, LEFT, RIGHT, FORWARD, BACKWARD, LAND.
     Stand where you will stand when flying (about 2-3 m from the camera), make the signal
     at chest or shoulder height, and press SPACE to capture a sample. Take 8-10 samples per
     signal; between samples move your hand a little, step a little closer or farther, turn
     slightly. ENTER moves on (after at least 5).
  2. Teach "relaxed hands": 15 s of moving naturally without signalling - hands at your sides,
     talking, scratching your head. This is what stops ordinary movement being read as a command.
  3. Check: every sample is compared with all the others. Signals that look too alike, or
     anything that could be mistaken for LAND or WAKE, is flagged and you can re-teach it.
  4. Live test: make each signal for 3 s and see how often it is recognized.
  5. False-trigger test: 20 s of natural movement; LAND and WAKE must never fire.
  6. Save. Only a profile that passed everything is marked "validated"; gesture_flight.py
     refuses an unvalidated profile unless you add --allow-unvalidated.

KEYS (same in the browser viewer):  SPACE = capture / start   ENTER = done / continue   Q = quit

TIPS FOR GOOD SIGNALS
  * Pick hand shapes that are very different from each other (finger count, fist vs open,
    pointing direction). Directions count: pointing left and pointing right are different.
  * LAND and WAKE should look like nothing else you do with your hands.
  * The profile stores numbers describing hand shapes, not pictures.

Requires: python3 -m pip install -U depthai opencv-python numpy mediapipe
"""

import argparse
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

import gesture_lib as gl
from gesture_lib import log

WINDOW = "Gesture setup  (q = quit)"
KEY_SPACE, KEY_Q = 32, ord("q")
KEYS_ENTER = (10, 13)
LIVE_TEST_S = 3.0
LIVE_PASS = 0.70
FALSE_TRIGGER_S = 20.0
NEUTRAL_S = 15.0
NEUTRAL_EVERY_S = 0.25


class Quit(Exception):
    pass


# ------------------------------------------------------------------ one camera frame + screen
class Session:
    def __init__(self, rig: gl.CameraRig, detector: gl.HandDetector, min_palm_px: float):
        self.rig, self.detector, self.min_palm_px = rig, detector, min_palm_px
        self.tracker = gl.PersonTracker()

    def step(self, clf: Optional[gl.GestureClassifier] = None):
        frame = self.rig.read(timeout=3.0)
        if frame is None:
            raise RuntimeError("No pictures from the camera (unplugged or stalled?)")
        tracked = self.tracker.update(frame.people, frame.t)
        subject = gl.nearest_person(list(tracked.values()))
        pid = next((tid for tid, p in tracked.items() if p is subject), None)
        if subject is None:
            res = gl.HandsResult(gl.NONE)
        else:
            res = gl.analyze_hands(frame, subject, self.detector, clf, self.min_palm_px)
        return frame, tracked, pid, subject, res

    def show(self, frame, tracked, pid, res, lines, color=(60, 60, 60), bottom=()) -> int:
        view = gl.draw_view(frame.bgr, tracked, pid, {pid: res} if pid is not None else {},
                            lines, color, bottom)
        cv2.imshow(WINDOW, view)
        key = cv2.waitKey(1) & 0xFF
        if key == KEY_Q:
            raise Quit()
        return key

    def wait_choice(self, lines, color=(60, 60, 60), bottom=(), keys=(KEY_SPACE,) + KEYS_ENTER) -> int:
        while True:
            frame, tracked, pid, _, res = self.step()
            key = self.show(frame, tracked, pid, res, lines, color, bottom)
            if key in keys:
                return key


def where_hint(subject: Optional[gl.Person], res: gl.HandsResult) -> str:
    if subject is None:
        return "No person in view - stand 1.5-3 m in front of the camera"
    if not res.hands and res.too_small:
        return "Hand too small in the picture - come a little closer"
    if not res.hands:
        return "No hand found - raise it to chest/shoulder height, inside the blue box"
    d = subject.distance_m
    return f"You are {d:.1f} m away" if d else "distance unknown"


def signal_hand(hands: List[gl.Hand]) -> Optional[gl.Hand]:
    """The hand making the signal: if two hands are visible, the higher one."""
    usable = [h for h in hands if h.feature is not None]
    if not usable:
        return None
    return min(usable, key=lambda h: h.wrist_y)


# ------------------------------------------------------------------ capturing
def capture_gesture(sess: Session, g: str, target: int, distances: List[float]) -> List[np.ndarray]:
    feats: List[np.ndarray] = []
    note = "Make the signal and press SPACE"
    log(f"TEACH {g.upper()}: {gl.SUGGESTIONS[g]}. SPACE = capture, ENTER = done, Q = quit")
    while True:
        frame, tracked, pid, subject, res = sess.step()
        lines = [f"Teach {g.upper()}:  {len(feats)}/{target} samples",
                 gl.SUGGESTIONS[g],
                 "SPACE = capture    ENTER = done (5+)    Q = quit",
                 note]
        key = sess.show(frame, tracked, pid, res, lines, bottom=[where_hint(subject, res)])
        if key == KEY_SPACE:
            hand = signal_hand(res.hands)
            if hand is None:
                note = where_hint(subject, res)
            else:
                feats.append(hand.feature)
                if subject is not None and subject.distance_m:
                    distances.append(subject.distance_m)
                extra = "  (2 hands seen - used the higher one)" if len(res.hands) > 1 else ""
                note = f"Captured {len(feats)}. Move/turn a little, then SPACE again.{extra}"
            log(note)
        elif key in KEYS_ENTER:
            if len(feats) >= gl.MIN_SAMPLES:
                break
            note = f"Need at least {gl.MIN_SAMPLES} samples (have {len(feats)})"
            log(note)
        if len(feats) >= target:
            break
    log(f"{g.upper()}: {len(feats)} samples")
    return feats


def capture_neutral(sess: Session) -> List[np.ndarray]:
    log("RELAXED HANDS: press SPACE, then move naturally for "
        f"{NEUTRAL_S:.0f} s without making any signal")
    sess.wait_choice(["Teach RELAXED HANDS (not a signal)",
                      f"Press SPACE, then for {NEUTRAL_S:.0f} s move naturally:",
                      "hands at your sides, talking, scratching your head, waving casually.",
                      "Do NOT make any of your signals."], keys=(KEY_SPACE,))
    feats: List[np.ndarray] = []
    t0 = last = time.monotonic()
    while time.monotonic() - t0 < NEUTRAL_S:
        frame, tracked, pid, subject, res = sess.step()
        now = time.monotonic()
        if now - last >= NEUTRAL_EVERY_S:
            last = now
            feats.extend(h.feature for h in res.hands if h.feature is not None)
        sess.show(frame, tracked, pid, res,
                  [f"RELAXED HANDS: keep moving naturally  {NEUTRAL_S - (now - t0):.0f} s",
                   f"{len(feats)} samples so far"], (120, 90, 0))
    log(f"Relaxed hands: {len(feats)} samples")
    return feats


# ------------------------------------------------------------------ checking
def problem_gestures(report: dict) -> List[str]:
    """Which gestures to re-teach, from the leave-one-out report."""
    out = []
    for g, acc in report["accuracy"].items():
        if acc < 0.7:
            out.append(g)
    for label, row in report["confusion"].items():
        for crit in (gl.LAND, gl.WAKE):
            if label != crit and row.get(crit, 0):
                out.append(crit)
                if label != gl.NEUTRAL:
                    out.append(label)
    for crit in report["critical"]:
        for g in gl.GESTURES:
            if crit.startswith(f"{g.upper()}: only"):
                out.append(g)
    return [g for g in gl.GESTURES if g in out]


def print_report(report: dict) -> None:
    log("---- CHECK: each sample compared with all the others ----")
    for g in gl.GESTURES:
        if g in report["accuracy"]:
            log(f"  {g.upper():9s} recognized {report['accuracy'][g]:.0%}   "
                f"read as: {report['confusion'].get(g, {})}")
    if gl.NEUTRAL in report["confusion"]:
        log(f"  RELAXED   read as: {report['confusion'][gl.NEUTRAL]}")
    for c in report["critical"]:
        log(f"  PROBLEM: {c}")
    for w in report["warnings"]:
        log(f"  warning: {w}")
    if not report["critical"]:
        log("  No critical problems.")


def live_test(sess: Session, clf: gl.GestureClassifier, gestures: List[str]) -> Dict[str, dict]:
    results = {}
    for g in gestures:
        sess.wait_choice([f"LIVE TEST {g.upper()}",
                          f"Make your {g.upper()} signal, then press SPACE and hold it {LIVE_TEST_S:.0f} s"],
                         keys=(KEY_SPACE,))
        counts: Dict[str, int] = {}
        t0 = time.monotonic()
        while time.monotonic() - t0 < LIVE_TEST_S:
            frame, tracked, pid, subject, res = sess.step(clf)
            if subject is not None:
                counts[res.label] = counts.get(res.label, 0) + 1
            sess.show(frame, tracked, pid, res,
                      [f"LIVE TEST {g.upper()}: hold it...  {LIVE_TEST_S - (time.monotonic() - t0):.1f} s",
                       f"now reading: {res.label.upper()}"],
                      (0, 140, 0) if res.label == g else (60, 60, 60))
        total = sum(counts.values())
        rate = counts.get(g, 0) / total if total else 0.0
        wrong = {k: v for k, v in counts.items() if k in gl.GESTURES and k != g}
        passed = rate >= LIVE_PASS and not any(k in (gl.LAND, gl.WAKE) for k in wrong)
        results[g] = {"rate": round(rate, 3), "frames": total, "counts": counts, "passed": passed}
        log(f"LIVE TEST {g.upper()}: recognized in {rate:.0%} of {total} frames "
            f"{'PASS' if passed else 'FAIL'}   ({counts})")
    return results


def false_trigger_test(sess: Session, clf: gl.GestureClassifier, timing: dict) -> dict:
    sess.wait_choice(["FALSE-TRIGGER TEST",
                      f"Press SPACE, then move naturally for {FALSE_TRIGGER_S:.0f} s.",
                      "Talk, gesture casually, move your arms - but make NONE of your signals."],
                     keys=(KEY_SPACE,))
    deb = gl.GestureDebouncer(timing)
    fired: Dict[str, int] = {}
    prev_active = None
    t0 = time.monotonic()
    while time.monotonic() - t0 < FALSE_TRIGGER_S:
        frame, tracked, pid, subject, res = sess.step(clf)
        now = time.monotonic()
        for ev in deb.update(now, res.label if subject is not None else gl.NONE):
            fired[ev] = fired.get(ev, 0) + 1
            log(f"  false trigger: {ev.upper()}")
        if deb.active and deb.active != prev_active:
            fired[deb.active] = fired.get(deb.active, 0) + 1
            log(f"  false trigger: {deb.active.upper()} (would have moved)")
        prev_active = deb.active
        sess.show(frame, tracked, pid, res,
                  [f"FALSE-TRIGGER TEST: move naturally  {FALSE_TRIGGER_S - (now - t0):.0f} s",
                   "triggers: " + (", ".join(f"{k.upper()} x{v}" for k, v in fired.items()) or "none")],
                  (0, 0, 160) if fired else (60, 60, 60))
    critical = {k: v for k, v in fired.items() if k in (gl.LAND, gl.WAKE)}
    moves = {k: v for k, v in fired.items() if k in gl.MOVES}
    passed = not critical
    log(f"FALSE-TRIGGER TEST: {'PASS' if passed else 'FAIL'}   land/wake: {critical or 'none'}   "
        f"movement: {moves or 'none'}")
    if moves:
        log("  (movement triggers are only warnings, but each would have moved the drone briefly)")
    return {"fired": fired, "passed": passed}


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Teach, check and save your hand signals.")
    p.add_argument("--name", default="operator", help="your name (stored in the profile)")
    p.add_argument("--profile", default="gesture_profile.json", help="profile file to write")
    p.add_argument("--samples", type=int, default=10, help="samples per signal (default 10)")
    p.add_argument("--redo", nargs="+", metavar="GESTURE",
                   help=f"re-teach only these ({', '.join(gl.GESTURES + [gl.NEUTRAL])}) in an existing profile")
    p.add_argument("--test-only", action="store_true", help="only run the live tests on the saved profile")
    p.add_argument("--skip-tests", action="store_true",
                   help="save without the live tests (profile is then NOT validated)")
    p.add_argument("--min-palm-px", type=float, default=10.0,
                   help="ignore hands smaller than this in the picture (pixels, wrist to knuckle)")
    cam = p.add_argument_group("camera")
    cam.add_argument("--fps", type=float, default=15)
    cam.add_argument("--frame-width", type=int, default=1280, help="width of the picture used for hands")
    cam.add_argument("--model", default="yolov6-nano", help="person detector (Luxonis zoo, COCO)")
    cam.add_argument("--conf", type=float, default=0.5, help="person detector confidence")
    cam.add_argument("--shaves", type=int, default=None, help="camera AI cores (default: auto)")
    return p.parse_args(argv)


def run(args) -> int:
    existing = None
    if args.redo or args.test_only:
        try:
            existing = gl.load_profile(args.profile)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"Can't use {args.profile}: {e}")
    if args.redo:
        bad = [g for g in args.redo if g not in gl.GESTURES + [gl.NEUTRAL]]
        if bad:
            sys.exit(f"Unknown gesture(s): {', '.join(bad)}. Choose from {', '.join(gl.GESTURES + [gl.NEUTRAL])}")

    samples: Dict[str, List[np.ndarray]] = {}
    name, timing, meta = args.name, dict(gl.DEFAULT_TIMING), {}
    if existing:
        for label, c in existing["classes"].items():
            samples[label] = [np.asarray(s, np.float32) for s in c.get("samples", [])]
        name = existing.get("name", name) if args.name == "operator" else args.name
        timing = existing["timing"]
        meta = existing.get("meta", {})

    if args.test_only:
        to_record: List[str] = []
    elif args.redo:
        to_record = [g for g in gl.GESTURES + [gl.NEUTRAL] if g in args.redo]
    else:
        to_record = gl.GESTURES + [gl.NEUTRAL]

    detector = gl.HandDetector(num_hands=2)
    rig = gl.CameraRig(args.model, args.fps, args.frame_width, args.conf, args.shaves)
    distances: List[float] = []
    try:
        rig.open()
        sess = Session(rig, detector, args.min_palm_px)
        while True:
            for g in to_record:
                if g == gl.NEUTRAL:
                    samples[g] = capture_neutral(sess)
                else:
                    samples[g] = capture_gesture(sess, g, max(args.samples, gl.MIN_SAMPLES), distances)
            if distances:
                meta["enroll_distance_m"] = round(float(np.median(distances)), 2)
            meta.update({"frame_width": args.frame_width, "model": args.model,
                         "min_palm_px": args.min_palm_px})
            profile = gl.build_profile(samples, name, timing, meta)
            report = gl.evaluate_profile(profile)
            profile["report"] = report
            print_report(report)

            if report["critical"]:
                redo = problem_gestures(report) or list(gl.GESTURES)
                key = sess.wait_choice(
                    ["CHECK FAILED: " + report["critical"][0],
                     f"SPACE = re-teach {', '.join(g.upper() for g in redo)}",
                     "ENTER = continue anyway (profile will NOT be validated)    Q = quit"],
                    (0, 0, 160), bottom=report["critical"][1:4])
                if key == KEY_SPACE:
                    to_record = redo
                    continue

            if args.skip_tests:
                profile["tests"] = {"skipped": True}
                break
            clf = gl.GestureClassifier(profile)
            live = live_test(sess, clf, list(gl.GESTURES))
            ft = false_trigger_test(sess, clf, profile["timing"])
            profile["tests"] = {"live": live, "false_trigger": ft,
                                "when": time.strftime("%Y-%m-%d %H:%M:%S")}
            failed = [g for g, r in live.items() if not r["passed"]]
            if not ft["passed"]:
                failed += [g for g in (gl.LAND, gl.WAKE) if ft["fired"].get(g) and g not in failed]
            if failed:
                key = sess.wait_choice(
                    ["TESTS FAILED: " + ", ".join(g.upper() for g in failed),
                     "SPACE = re-teach those and test again",
                     "ENTER = save anyway (NOT validated)    Q = quit"], (0, 0, 160))
                if key == KEY_SPACE:
                    to_record = failed
                    continue
            break
    except Quit:
        log("Quit - nothing saved.")
        return 1
    finally:
        cv2.destroyAllWindows()
        rig.close()
        detector.close()

    tests = profile.get("tests", {})
    passed_tests = (not tests.get("skipped") and all(r["passed"] for r in tests.get("live", {}).values())
                    and tests.get("false_trigger", {}).get("passed", False))
    profile["validated"] = bool(not profile["report"]["critical"] and passed_tests)
    gl.save_profile(args.profile, profile)
    log(f"Saved {args.profile} for '{name}'  -  "
        + ("VALIDATED, ready for gesture_flight.py" if profile["validated"]
           else "NOT validated (gesture_flight.py needs --allow-unvalidated)"))
    return 0


def main():
    args = parse_args()
    if not gl.display_available(False):
        sys.exit("The setup needs to show you the camera picture. With no monitor, run:\n"
                 "  python3 remote_view.py gesture_setup.py " + " ".join(sys.argv[1:]))
    try:
        sys.exit(run(args))
    except KeyboardInterrupt:
        log("Ctrl+C - nothing saved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
