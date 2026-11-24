#!/usr/bin/env python3
"""
test_gesture_logic.py - checks the gesture code WITHOUT a camera or a drone.

A fake camera and fake hands stand in for the OAK-D and MediaPipe, so you can run this on
the Jetson or on any computer after changing the code:

    python3 test_gesture_logic.py          # about 1-2 minutes

It runs the real setup flow (teach, check, tests, save) and a full simulated flight
(recheck, takeoff, WAKE, moves, LAND), plus the safety rules one by one.
"""

import asyncio
import io
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gesture_lib as gl          # noqa: E402
import gesture_flight as gf       # noqa: E402
import gesture_setup as gs        # noqa: E402

ALL = gl.GESTURES + [gl.NEUTRAL]


# ------------------------------------------------------------------ fakes
def make_prototypes(seed=7):
    """A random but fixed hand shape (21 points, wrist at 0, palm length 1) per gesture."""
    rng = np.random.default_rng(seed)
    protos = {}
    for label in ALL:
        pts = np.zeros((21, 3), np.float32)
        pts[1:, :2] = rng.uniform(-2.0, 2.0, (20, 2))
        pts[1:, 2] = rng.uniform(-0.5, 0.5, 20)
        v = pts[9, :2] / np.linalg.norm(pts[9, :2])
        pts[9, :2] = v          # wrist-to-middle-knuckle length = 1
        protos[label] = pts
    return protos


PROTOS = make_prototypes()


def noisy_landmarks(label, rng, palm_px=30.0, at=(100.0, 100.0), sigma=0.05):
    pts = PROTOS[label].copy()
    pts[1:] += rng.normal(0, sigma, (20, 3)).astype(np.float32)
    lm = pts * palm_px
    lm[:, 0] += at[0]
    lm[:, 1] += at[1]
    return lm


def fake_samples(n=10, n_neutral=30, seed=1):
    rng = np.random.default_rng(seed)
    out = {}
    for label in ALL:
        k = n_neutral if label == gl.NEUTRAL else n
        out[label] = [gl.hand_features(noisy_landmarks(label, rng)) for _ in range(k)]
    return out


def fake_profile(validated=True):
    prof = gl.build_profile(fake_samples(), "tester")
    prof["report"] = gl.evaluate_profile(prof)
    prof["validated"] = validated
    return prof


class FakeHands:
    """Stands in for gl.HandDetector. `chooser()` returns the gesture to show (or None)."""

    def __init__(self, chooser, seed=3, palm_px=30.0):
        self.chooser, self.rng, self.palm_px = chooser, np.random.default_rng(seed), palm_px
        self.calls = 0

    def detect(self, bgr):
        self.calls += 1
        g = self.chooser()
        if g is None:
            return []
        h, w = bgr.shape[:2]
        return [(noisy_landmarks(g, self.rng, self.palm_px, (w / 2, h / 2)), "Right", 0.9)]

    def close(self):
        pass


class FakeRig:
    """Stands in for gl.CameraRig: one person 2.5 m straight ahead, depth 2.5 m everywhere."""

    def __init__(self, fps=15, dist_mm=2500.0, people=None):
        self.fps, self.dist_mm = fps, dist_mm
        self.people = people
        self.running = True
        self.stall = False
        self.bgr = np.zeros((480, 640, 3), np.uint8)
        self.depth = np.full((400, 640), int(dist_mm), np.uint16)

    def open(self):
        return self

    def is_running(self):
        return self.running

    def read(self, timeout=1.0):
        time.sleep(1.0 / self.fps)
        if self.stall or not self.running:
            return None
        people = self.people if self.people is not None else [
            gl.Person((0.40, 0.10, 0.60, 0.95), 0.9, 0.0, 0.0, self.dist_mm)]
        return gl.Frame(time.monotonic(), self.bgr, list(people), self.depth)

    def close(self):
        self.running = False


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = fn(*a, **k)
    return result, buf.getvalue()


# ------------------------------------------------------------------ unit tests
class TestFeaturesAndClassifier(unittest.TestCase):
    def test_features_ignore_position_and_size(self):
        rng = np.random.default_rng(0)
        lm = noisy_landmarks(gl.LEFT, rng, sigma=0)
        moved = lm.copy()
        moved[:, :2] = lm[:, :2] * 2.5 + 400        # bigger and elsewhere (z scales too)
        moved[:, 2] = lm[:, 2] * 2.5
        np.testing.assert_allclose(gl.hand_features(lm), gl.hand_features(moved), atol=1e-4)

    def test_features_keep_direction(self):
        rng = np.random.default_rng(0)
        lm = noisy_landmarks(gl.LEFT, rng, sigma=0)
        mirrored = lm.copy()
        mirrored[:, 0] = -mirrored[:, 0]
        self.assertGreater(np.linalg.norm(gl.hand_features(lm) - gl.hand_features(mirrored)), 1.0)

    def test_degenerate_hand(self):
        self.assertIsNone(gl.hand_features(np.zeros((21, 3), np.float32)))

    def test_classifier_recognizes_and_rejects(self):
        prof = fake_profile()
        clf = gl.GestureClassifier(prof)
        rng = np.random.default_rng(99)
        for g in gl.GESTURES:
            label, _, _ = clf.classify(gl.hand_features(noisy_landmarks(g, rng)))
            self.assertEqual(label, g)
        self.assertEqual(clf.classify(gl.hand_features(noisy_landmarks(gl.NEUTRAL, rng)))[0], gl.NEUTRAL)
        # a hand shape that was never taught -> not a command
        other = np.random.default_rng(1234).uniform(-2, 2, 60).astype(np.float32)
        self.assertIn(clf.classify(other)[0], gl.PASSIVE)

    def test_halfway_between_two_signals_is_rejected(self):
        clf = gl.GestureClassifier(fake_profile())
        mid = (gl.hand_features(noisy_landmarks(gl.LEFT, np.random.default_rng(1), sigma=0)) +
               gl.hand_features(noisy_landmarks(gl.LAND, np.random.default_rng(1), sigma=0))) / 2
        self.assertEqual(clf.classify(mid)[0], gl.UNKNOWN)

    def test_good_profile_has_no_critical_problems(self):
        rep = fake_profile()["report"]
        self.assertEqual(rep["critical"], [])
        self.assertTrue(all(a >= 0.9 for a in rep["accuracy"].values()))

    def test_land_that_looks_like_left_is_critical(self):
        samples = fake_samples()
        rng = np.random.default_rng(5)
        samples[gl.LAND] = [gl.hand_features(noisy_landmarks(gl.LEFT, rng)) for _ in range(10)]
        rep = gl.evaluate_profile(gl.build_profile(samples, "x"))
        self.assertTrue(rep["critical"])
        self.assertIn(gl.LAND, gs.problem_gestures(rep))
        self.assertIn(gl.LEFT, gs.problem_gestures(rep))

    def test_too_few_samples_is_critical(self):
        samples = fake_samples()
        samples[gl.WAKE] = samples[gl.WAKE][:3]
        rep = gl.evaluate_profile(gl.build_profile(samples, "x"))
        self.assertTrue(any("WAKE: only 3" in c for c in rep["critical"]))
        self.assertIn(gl.WAKE, gs.problem_gestures(rep))

    def test_no_neutral_samples_still_works(self):
        samples = fake_samples()
        samples[gl.NEUTRAL] = []
        prof = gl.build_profile(samples, "x")
        rep = gl.evaluate_profile(prof)
        self.assertEqual(rep["critical"], [])
        self.assertTrue(any("relaxed-hand" in w for w in rep["warnings"]))
        gl.GestureClassifier(prof)

    def test_profile_round_trip(self):
        prof = fake_profile()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.json")
            gl.save_profile(p, prof)
            back = gl.load_profile(p)
            json.dumps(back)
        f = gl.hand_features(noisy_landmarks(gl.RIGHT, np.random.default_rng(8)))
        self.assertEqual(gl.GestureClassifier(back).classify(f)[0], gl.RIGHT)

    def test_load_rejects_incomplete_profile(self):
        prof = fake_profile()
        del prof["classes"][gl.LAND]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "p.json")
            gl.save_profile(p, prof)
            with self.assertRaises(ValueError):
                gl.load_profile(p)

    def test_combine_labels(self):
        self.assertEqual(gl.combine_labels([]), gl.NONE)
        self.assertEqual(gl.combine_labels([gl.LEFT, gl.NEUTRAL]), gl.LEFT)
        self.assertEqual(gl.combine_labels([gl.LEFT, gl.LAND]), gl.CONFLICT)
        self.assertEqual(gl.combine_labels([gl.LEFT, gl.LEFT]), gl.LEFT)
        self.assertEqual(gl.combine_labels([gl.UNKNOWN, gl.NEUTRAL]), gl.NEUTRAL)
        self.assertEqual(gl.combine_labels([gl.UNKNOWN]), gl.UNKNOWN)


class TestDebouncer(unittest.TestCase):
    def feed(self, deb, plan, fps=15, t0=0.0):
        """plan: list of (label, seconds). Returns (events with times, active per frame)."""
        t, events, actives = t0, [], []
        for label, dur in plan:
            for _ in range(int(round(dur * fps))):
                t += 1.0 / fps
                for e in deb.update(t, label):
                    events.append((round(t, 2), e))
                actives.append((round(t, 2), deb.active))
        return events, actives

    def test_hold_to_move_start_and_release(self):
        deb = gl.GestureDebouncer(gl.DEFAULT_TIMING)
        _, act = self.feed(deb, [(gl.NEUTRAL, 1.0), (gl.LEFT, 1.0), (gl.NEUTRAL, 1.0)])
        on = [t for t, a in act if a == gl.LEFT]
        self.assertTrue(on)
        self.assertAlmostEqual(on[0] - 1.0, 0.4, delta=0.1)     # starts after start_s
        self.assertAlmostEqual(on[-1] - 2.0, 0.3, delta=0.1)    # stops release_s after
        self.assertIsNone(act[-1][1])

    def test_brief_flicker_does_not_start(self):
        deb = gl.GestureDebouncer(gl.DEFAULT_TIMING)
        _, act = self.feed(deb, [(gl.NEUTRAL, 1.0), (gl.LEFT, 0.2), (gl.NEUTRAL, 1.0)] * 3)
        self.assertFalse(any(a for _, a in act))

    def test_single_dropout_keeps_moving(self):
        deb = gl.GestureDebouncer(gl.DEFAULT_TIMING)
        plan = [(gl.NEUTRAL, 1.0), (gl.LEFT, 1.0), (gl.NONE, 1 / 15), (gl.LEFT, 1.0)]
        _, act = self.feed(deb, plan)
        moving = [a for t, a in act if 1.5 < t < 3.0]
        self.assertTrue(all(a == gl.LEFT for a in moving))

    def test_other_signal_stops_at_once(self):
        deb = gl.GestureDebouncer(gl.DEFAULT_TIMING)
        _, act = self.feed(deb, [(gl.NEUTRAL, 1.0), (gl.LEFT, 1.0), (gl.CONFLICT, 1 / 15)])
        self.assertIsNone(act[-1][1])

    def test_land_needs_full_hold_and_fires_once(self):
        deb = gl.GestureDebouncer(gl.DEFAULT_TIMING)
        ev, _ = self.feed(deb, [(gl.NEUTRAL, 1.0), (gl.LAND, 1.0), (gl.NEUTRAL, 1.0)])
        self.assertEqual(ev, [])
        ev, _ = self.feed(deb, [(gl.LAND, 3.0)], t0=10.0)
        self.assertEqual([e for _, e in ev], [gl.LAND])
        self.assertAlmostEqual(ev[0][0] - 10.0, 1.5, delta=0.15)
        ev, _ = self.feed(deb, [(gl.NEUTRAL, 1.0), (gl.LAND, 2.0)], t0=13.0)
        self.assertEqual([e for _, e in ev], [gl.LAND])     # re-armed after letting go

    def test_land_with_some_missed_frames(self):
        deb = gl.GestureDebouncer(gl.DEFAULT_TIMING)
        plan = [(gl.NEUTRAL, 1.0)] + [(gl.LAND, 0.4), (gl.UNKNOWN, 1 / 15)] * 5
        ev, _ = self.feed(deb, plan)
        self.assertIn(gl.LAND, [e for _, e in ev])

    def test_land_does_not_move(self):
        deb = gl.GestureDebouncer(gl.DEFAULT_TIMING)
        _, act = self.feed(deb, [(gl.LAND, 2.0), (gl.WAKE, 2.0)])
        self.assertFalse(any(a for _, a in act))


class TestTrackerCropDepth(unittest.TestCase):
    def test_tracker_keeps_id_and_forgets(self):
        tr = gl.PersonTracker(max_age=1.0)
        a = gl.Person((0.4, 0.1, 0.6, 0.9), 0.9, 0, 0, 2500)
        b = gl.Person((0.05, 0.1, 0.2, 0.9), 0.9, 0, 0, 3000)
        ids1 = tr.update([a, b], 0.0)
        a2 = gl.Person((0.42, 0.1, 0.62, 0.9), 0.9, 0, 0, 2550)
        ids2 = tr.update([b, a2], 0.1)
        id_a = next(i for i, p in ids1.items() if p is a)
        self.assertIs(ids2[id_a], a2)
        tr.update([b], 0.6)
        self.assertTrue(tr.alive(id_a))
        tr.update([b], 1.2)
        self.assertFalse(tr.alive(id_a))
        ids3 = tr.update([a, b], 1.3)
        self.assertNotIn(id_a, ids3)          # came back -> new number

    def test_tracker_rejects_depth_jump(self):
        tr = gl.PersonTracker()
        near = gl.Person((0.4, 0.1, 0.6, 0.9), 0.9, 0, 0, 2000)
        far = gl.Person((0.4, 0.1, 0.6, 0.9), 0.9, 0, 0, 5000)
        i1 = tr.update([near], 0.0)
        i2 = tr.update([far], 0.1)
        self.assertNotEqual(set(i1), set(i2))

    def test_crop(self):
        p = gl.Person((0.4, 0.1, 0.6, 0.9), 0.9, 0, 0, 2500)
        x0, y0, x1, y1 = gl.person_crop(p, (960, 1280))
        self.assertLess(x0, 0.4 * 1280)
        self.assertGreater(x1, 0.6 * 1280)
        self.assertLess(y1, 0.9 * 960)
        self.assertIsNone(gl.person_crop(gl.Person((0.5, 0.5, 0.51, 0.51), 1, 0, 0, 0), (960, 1280)))

    def test_analyze_hands_maps_to_frame_coords(self):
        clf = gl.GestureClassifier(fake_profile())
        frame = gl.Frame(0.0, np.zeros((960, 1280, 3), np.uint8),
                         [gl.Person((0.4, 0.1, 0.6, 0.9), 0.9, 0, 0, 2500)], None)
        res = gl.analyze_hands(frame, frame.people[0], FakeHands(lambda: gl.RIGHT), clf)
        self.assertEqual(res.label, gl.RIGHT)
        x0, y0, x1, y1 = res.crop
        wx, wy = res.hands[0].lm[0, :2]
        self.assertTrue(x0 <= wx <= x1 and y0 <= wy <= y1)
        small = gl.analyze_hands(frame, frame.people[0], FakeHands(lambda: gl.RIGHT, palm_px=5), clf)
        self.assertEqual((small.label, small.too_small), (gl.NONE, 1))

    def test_depth_rules(self):
        self.assertTrue(gl.analyze_depth(None, 1.2)[1])
        self.assertTrue(gl.analyze_depth(np.zeros((400, 640), np.uint16), 1.2)[1])
        self.assertTrue(gl.analyze_depth(np.full((400, 640), 900, np.uint16), 1.2)[1])
        d, blocked, _ = gl.analyze_depth(np.full((400, 640), 2500, np.uint16), 1.2)
        self.assertFalse(blocked)
        self.assertAlmostEqual(d, 2.5, places=2)

    def test_draw_view_runs(self):
        frame = np.zeros((960, 1280, 3), np.uint8)
        p = gl.Person((0.4, 0.1, 0.6, 0.9), 0.9, 0, 0, 2500)
        clf = gl.GestureClassifier(fake_profile())
        res = gl.analyze_hands(gl.Frame(0, frame, [p], None), p, FakeHands(lambda: gl.LAND), clf)
        v = gl.draw_view(frame, {1: p}, 1, {1: res}, ["a", "b"], bottom_lines=["c"])
        self.assertEqual(v.shape[1], gl.VIEW_WIDTH)


class TestDecisions(unittest.TestCase):
    def args(self, *extra):
        return gf.parse_args(list(extra))

    def test_direction_frames(self):
        self.assertEqual(gf.velocity_for(gl.LEFT, "operator", 0.5, 0.3), (0.0, 0.3))
        self.assertEqual(gf.velocity_for(gl.RIGHT, "operator", 0.5, 0.3), (-0.0, -0.3))
        self.assertEqual(gf.velocity_for(gl.BACKWARD, "operator", 0.5, 0.3), (0.5, 0.0))
        self.assertEqual(gf.velocity_for(gl.FORWARD, "operator", 0.5, 0.3), (-0.3, 0.0))
        self.assertEqual(gf.velocity_for(gl.FORWARD, "drone", 0.5, 0.3), (0.5, 0.0))
        self.assertEqual(gf.velocity_for(gl.LEFT, "drone", 0.5, 0.3), (0.0, -0.3))

    def snap(self, **kw):
        base = dict(frame_time=time.monotonic(), operator_id=1, operator_dist_m=2.5, operator_seen=time.monotonic(),
                    active=gl.BACKWARD, active_since=time.monotonic(), depth_m=2.5, depth_blocked=False,
                    depth_reason="CLEAR")
        base.update(kw)
        return gf.Snapshot(**base)

    def test_blocked_reasons(self):
        a, cmd, now = self.args(), gf.SimCommander(), time.monotonic()
        self.assertIsNone(gf.blocked_reason(self.snap(), 0.5, 0, now, cmd, a))
        self.assertIn("too close", gf.blocked_reason(self.snap(operator_dist_m=1.5), 0.5, 0, now, cmd, a))
        self.assertIn("unknown", gf.blocked_reason(self.snap(operator_dist_m=None), 0.5, 0, now, cmd, a))
        self.assertIn("depth", gf.blocked_reason(self.snap(depth_blocked=True, depth_reason="OBSTACLE", depth_m=0.8),
                                                 0.5, 0, now, cmd, a))
        # depth/distance rules only apply toward the camera's view
        self.assertIsNone(gf.blocked_reason(self.snap(operator_dist_m=1.5, depth_blocked=True), -0.3, 0, now, cmd, a))
        self.assertIn("held more", gf.blocked_reason(self.snap(active_since=now - 5), 0.5, 0, now, cmd, a))

    def test_fence(self):
        a, cmd, now = self.args("--fence", "3"), gf.SimCommander(), time.monotonic()
        cmd._pos[:] = (0.0, 2.95)          # east of takeoff, heading north
        self.assertIn("fence", gf.blocked_reason(self.snap(), 0.0, 0.3, now, cmd, a))   # right = east = out
        self.assertIsNone(gf.blocked_reason(self.snap(), 0.0, -0.3, now, cmd, a))       # back toward home
        cmd._heading = 90.0                # facing east: forward is out
        self.assertIn("fence", gf.blocked_reason(self.snap(), 0.5, 0.0, now, cmd, a))

    def test_yaw(self):
        self.assertEqual(gf.yaw_for(self.snap(operator_bearing_deg=3)), 0.0)
        self.assertEqual(gf.yaw_for(self.snap(operator_bearing_deg=100)), gf.YAW_MAX)
        self.assertEqual(gf.yaw_for(self.snap(operator_bearing_deg=-10)), -5.0)

    def test_arg_rules(self):
        with self.assertRaises(SystemExit):
            quiet(gf.parse_args, ["--fly"])
        with self.assertRaises(SystemExit):
            quiet(gf.parse_args, ["--live", "--skip-recheck"])
        with self.assertRaises(SystemExit):
            quiet(gf.parse_args, ["--speed", "3"])


# ------------------------------------------------------------------ control loop with a scripted perception
class ScriptedPerception:
    """Minimal stand-in for gf.Perception; tests edit `.s` while the loop runs."""

    def __init__(self):
        now = time.monotonic()
        self.s = gf.Snapshot(frame_time=now, depth_m=2.5, depth_blocked=False, depth_reason="CLEAR")
        self.allow_lock = False
        self.banner = []
        self._alive = True
        threading.Thread(target=self._tick, daemon=True).start()

    def _tick(self):
        while self._alive:
            if not getattr(self, "frozen", False):
                self.s = gf.replace(self.s, frame_time=time.monotonic(),
                                    operator_seen=time.monotonic() if self.s.operator_id else self.s.operator_seen)
            time.sleep(0.05)

    def snapshot(self):
        return self.s

    def set_banner(self, lines, color=None):
        self.banner = list(lines)


class LoggingSim(gf.SimCommander):
    def __init__(self):
        super().__init__()
        self.cmds = []

    async def set_velocity(self, fwd, right, yaw):
        self.cmds.append((time.monotonic(), round(fwd, 2), round(right, 2), round(yaw, 1)))
        await super().set_velocity(fwd, right, yaw)


def run_loop(perc, args, script, limit=20.0):
    """Run control_loop while `script(perc, t)` edits the perception; returns (reason, commands)."""
    cmd = LoggingSim()
    cmd._t = time.monotonic()

    async def go():
        abort = asyncio.Event()

        async def driver():
            t0 = time.monotonic()
            while True:
                script(perc, time.monotonic() - t0, abort)
                await asyncio.sleep(0.02)

        d = asyncio.create_task(driver())
        try:
            return await asyncio.wait_for(gf.control_loop(cmd, perc, args, abort), limit)
        finally:
            d.cancel()

    try:
        reason, out = quiet(asyncio.run, go())
    finally:
        perc._alive = False
    return reason, cmd.cmds, out


class TestControlLoop(unittest.TestCase):
    def args(self, *extra):
        return gf.parse_args(["--wake-timeout", "1.5", "--lost-land", "1.0", "--stale-land", "0.8"] + list(extra))

    def test_no_wake_lands(self):
        reason, cmds, _ = run_loop(ScriptedPerception(), self.args(), lambda p, t, a: None)
        self.assertEqual(reason, "no WAKE signal")
        self.assertTrue(all(c[1:] == (0, 0, 0) for c in cmds))

    def test_move_then_land(self):
        def script(p, t, a):
            if t > 0.2 and p.s.operator_id is None:
                p.s = gf.replace(p.s, operator_id=4, operator_dist_m=2.6, operator_seen=time.monotonic())
            if 0.5 < t < 1.0:
                p.s = gf.replace(p.s, active=gl.LEFT, active_since=p.s.active_since or time.monotonic())
            elif 1.0 <= t < 1.5:
                p.s = gf.replace(p.s, active=None, active_since=0.0)
            if t > 1.6:
                p.s = gf.replace(p.s, land_events=1)
        reason, cmds, _ = run_loop(ScriptedPerception(), self.args(), script)
        self.assertEqual(reason, "LAND signal")
        moving = [c for c in cmds if c[2] != 0]
        self.assertTrue(moving)
        self.assertTrue(all(c[2] == 0.3 and c[1] == 0 for c in moving))   # LEFT = drone's right in operator frame
        self.assertEqual(cmds[-1][1:], (0, 0, 0))

    def test_operator_lost_lands(self):
        def script(p, t, a):
            if 0.2 < t < 0.6:
                p.s = gf.replace(p.s, operator_id=4, operator_seen=time.monotonic())
            elif t >= 0.6 and p.s.operator_id:
                p.s = gf.replace(p.s, operator_id=None, lost_time=time.monotonic())
        reason, _, out = run_loop(ScriptedPerception(), self.args(), script)
        self.assertEqual(reason, "operator lost")
        self.assertIn("OPERATOR LOST", out)

    def test_rewake_after_loss_continues(self):
        def script(p, t, a):
            if 0.2 < t < 0.5:
                p.s = gf.replace(p.s, operator_id=4)
            elif 0.5 <= t < 1.0 and p.s.operator_id == 4:
                p.s = gf.replace(p.s, operator_id=None, lost_time=time.monotonic())
            elif 1.0 <= t < 2.0 and p.s.operator_id is None:
                p.s = gf.replace(p.s, operator_id=5, operator_seen=time.monotonic())
            elif t >= 2.5:
                a.set()
        reason, _, _ = run_loop(ScriptedPerception(), self.args(), script)
        self.assertEqual(reason, "operator abort (q / Ctrl+C)")

    def test_camera_stall_lands(self):
        def script(p, t, a):
            if t > 0.3:
                p.frozen = True
        reason, cmds, _ = run_loop(ScriptedPerception(), self.args(), script)
        self.assertEqual(reason, "camera stalled")
        self.assertTrue(all(c[1:] == (0, 0, 0) for c in cmds))

    def test_refused_move_hovers(self):
        def script(p, t, a):
            if t > 0.1 and p.s.operator_id is None:
                p.s = gf.replace(p.s, operator_id=4, operator_dist_m=1.6, active=gl.BACKWARD,
                                 active_since=time.monotonic())
            if t > 1.0:
                a.set()
        reason, cmds, out = run_loop(ScriptedPerception(), self.args(), script)
        self.assertTrue(all(c[1:] == (0, 0, 0) for c in cmds))
        self.assertIn("refused: too close", out)

    def test_operator_unseen_hovers(self):
        def script(p, t, a):
            if t > 0.1 and p.s.operator_id is None:
                p.s = gf.replace(p.s, operator_id=4, active=gl.LEFT, active_since=time.monotonic())
                p.frozen_seen = True
            if t > 1.2:
                a.set()

        class P(ScriptedPerception):
            def _tick(self):
                while self._alive:      # frames keep coming, but the operator box stops updating
                    self.s = gf.replace(self.s, frame_time=time.monotonic())
                    time.sleep(0.05)
        perc = P()
        perc.s = gf.replace(perc.s, operator_seen=time.monotonic())
        reason, cmds, out = run_loop(perc, self.args(), script)
        late = [c for c in cmds if c[0] > cmds[0][0] + 0.8]
        self.assertTrue(late and all(c[1:] == (0, 0, 0) for c in late))
        self.assertIn("operator not visible", out)


# ------------------------------------------------------------------ the real Perception with fake camera/hands
class TestPerception(unittest.TestCase):
    def make(self, chooser, people=None, allow_lock=True):
        args = gf.parse_args([])
        prof = fake_profile()
        perc = gf.Perception(FakeRig(people=people), FakeHands(chooser), gl.GestureClassifier(prof),
                             prof["timing"], args, show=True)
        perc.allow_lock = allow_lock
        return perc

    def run_frames(self, perc, plan, fps=15):
        """plan: list of (seconds, gesture-or-None); drives perc.process directly (no thread)."""
        cur = {"g": None}
        perc.detector.chooser = lambda: cur["g"]
        t = getattr(perc, "_fake_t", 100.0)
        rig = perc.source
        for dur, g in plan:
            cur["g"] = g
            for _ in range(int(dur * fps)):
                t += 1.0 / fps
                people = rig.people if rig.people is not None else [
                    gl.Person((0.40, 0.10, 0.60, 0.95), 0.9, 0.0, 0.0, rig.dist_mm)]
                f = gl.Frame(t, rig.bgr, list(people), rig.depth)
                perc.process(f)
        perc._fake_t = t
        return perc.snapshot()

    def test_wake_locks_and_land_counts(self):
        perc = self.make(None)
        s, out = quiet(self.run_frames, perc, [(1.0, gl.NEUTRAL), (1.4, gl.WAKE)])
        self.assertEqual(s.operator_id, 1)
        self.assertIn("WAKE: operator locked", out)
        s, _ = quiet(self.run_frames, perc, [(0.5, gl.NEUTRAL), (1.0, gl.RIGHT)])
        self.assertEqual(s.active, gl.RIGHT)
        s, out = quiet(self.run_frames, perc, [(0.5, gl.NEUTRAL), (2.0, gl.LAND)])
        self.assertEqual(s.land_events, 1)
        self.assertIsNone(s.active)
        self.assertIsNotNone(perc.latest_view())

    def test_no_lock_when_not_allowed(self):
        perc = self.make(None, allow_lock=False)
        s, _ = quiet(self.run_frames, perc, [(2.0, gl.WAKE)])
        self.assertIsNone(s.operator_id)
        self.assertIn(gl.WAKE, s.held)          # recheck still sees it

    def test_operator_leaves_and_is_forgotten(self):
        perc = self.make(None)
        quiet(self.run_frames, perc, [(1.5, gl.WAKE)])
        self.assertEqual(perc.snapshot().operator_id, 1)
        perc.source.people = []
        s, out = quiet(self.run_frames, perc, [(1.5, None)])
        self.assertIsNone(s.operator_id)
        self.assertGreater(s.lost_time, 0)
        self.assertIn("LOST", out)

    def test_far_person_cannot_wake(self):
        far = [gl.Person((0.40, 0.10, 0.60, 0.95), 0.9, 0.0, 0.0, 6000)]
        perc = self.make(None, people=far)
        s, _ = quiet(self.run_frames, perc, [(2.0, gl.WAKE)])
        self.assertIsNone(s.operator_id)

    def test_bystander_ignored_after_lock(self):
        op = gl.Person((0.40, 0.10, 0.60, 0.95), 0.9, 0.0, 0.0, 2500)
        other = gl.Person((0.05, 0.10, 0.25, 0.95), 0.9, -900.0, 0.0, 2800)
        perc = self.make(None, people=[op, other])
        quiet(self.run_frames, perc, [(1.5, gl.WAKE)])
        locked = perc.snapshot().operator_id
        self.assertIsNotNone(locked)
        calls = perc.detector.calls
        quiet(self.run_frames, perc, [(1.0, gl.NEUTRAL)])
        self.assertEqual(perc.detector.calls - calls, 15)    # one hand check per frame: operator only

    def test_thread_reports_camera_failure(self):
        perc = self.make(lambda: gl.NEUTRAL)
        perc.start()
        time.sleep(0.5)
        self.assertGreater(perc.snapshot().frame_time, 0)
        _, out = quiet(lambda: (perc.source.close(), time.sleep(1.2)))
        self.assertEqual(perc.snapshot().error, "camera stopped")
        perc.stop()


# ------------------------------------------------------------------ whole programs
class Screen:
    """Captures the top banner line drawn by gl.draw_view so fake hands can 'read' prompts."""

    def __init__(self):
        self.line = ""
        self._orig = gl.draw_view

    def __enter__(self):
        def spy(frame, tracked, hid, results, top_lines, *a, **k):
            self.line = top_lines[0] if top_lines else ""
            return self._orig(frame, tracked, hid, results, top_lines, *a, **k)
        gl.draw_view = spy
        return self

    def __exit__(self, *e):
        gl.draw_view = self._orig


def gesture_on_screen(line: str):
    up = line.upper()
    if "RELAXED" in up or "FALSE-TRIGGER" in up:
        return gl.NEUTRAL
    for g in gl.GESTURES:
        if f"{g.upper()}:" in up or f"TEST {g.upper()}" in up or f"SHOW {g.upper()}" in up:
            return g
    return gl.NEUTRAL


class TestWholePrograms(unittest.TestCase):
    def test_setup_flow_saves_validated_profile(self):
        import cv2
        orig = (cv2.imshow, cv2.waitKey, cv2.destroyAllWindows)
        n = {"i": 0}

        def waitkey(_=0):
            n["i"] += 1
            return 32 if n["i"] % 3 == 0 else -1     # press SPACE every 3rd frame

        with tempfile.TemporaryDirectory() as d, Screen() as screen:
            prof_path = os.path.join(d, "g.json")
            cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = (lambda *a: None), waitkey, (lambda: None)
            gs.NEUTRAL_S, gs.FALSE_TRIGGER_S, gs.LIVE_TEST_S = 2.0, 3.0, 1.0
            orig_rig, orig_hd = gl.CameraRig, gl.HandDetector
            gl.CameraRig = lambda *a, **k: FakeRig(fps=30)
            gl.HandDetector = lambda **k: FakeHands(lambda: gesture_on_screen(screen.line))
            try:
                rc, out = quiet(gs.run, gs.parse_args(["--name", "Tester", "--profile", prof_path, "--samples", "6"]))
            finally:
                cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = orig
                gl.CameraRig, gl.HandDetector = orig_rig, orig_hd
            self.assertEqual(rc, 0, out[-2000:])
            prof = gl.load_profile(prof_path)
        self.assertTrue(prof["validated"], out[-3000:])
        self.assertEqual(prof["name"], "Tester")
        self.assertEqual(len(prof["classes"][gl.LAND]["samples"]), 6)
        self.assertGreater(len(prof["classes"][gl.NEUTRAL]["samples"]), 5)
        self.assertTrue(prof["tests"]["false_trigger"]["passed"])
        self.assertIn("VALIDATED", out)

    def test_full_simulated_flight(self):
        args = gf.parse_args(["--no-display"])
        prof = fake_profile()
        phase = {"t0": None}

        def chooser():
            line = " ".join(perc.banner)
            if "RECHECK" in line:
                return gesture_on_screen(line)
            if "hold WAKE to take control" in line and phase["t0"] is None:
                phase["t0"] = time.monotonic()
            if phase["t0"] is None:
                return gl.NEUTRAL
            t = time.monotonic() - phase["t0"]
            for end, g in ((1.0, gl.NEUTRAL), (2.5, gl.WAKE), (3.0, gl.NEUTRAL), (4.5, gl.LEFT),
                           (5.0, None), (6.5, gl.BACKWARD), (7.0, gl.NEUTRAL), (9.5, gl.LAND)):
                if t < end:
                    return g
            return gl.NEUTRAL

        perc = gf.Perception(FakeRig(), FakeHands(chooser), gl.GestureClassifier(prof), prof["timing"],
                             args, show=False)
        perc.start()
        try:
            _, out = quiet(asyncio.run, gf.main_async(args, perc, False))
        finally:
            perc.stop()
        self.assertIn("RECHECK passed", out, out[-2500:])
        self.assertIn("WAKE: operator locked", out)
        self.assertIn("right +0.30 m/s", out)          # LEFT (your left) = drone's right
        self.assertIn("forward +0.50 m/s", out)        # BACKWARD (toward you) = drone forward
        self.assertIn("Ending flight: LAND signal", out)
        self.assertIn("[SIM] landed", out)

    def test_flight_refuses_when_recheck_fails(self):
        args = gf.parse_args(["--no-display", "--recheck-timeout", "1.5"])
        prof = fake_profile()
        perc = gf.Perception(FakeRig(), FakeHands(lambda: gl.NEUTRAL), gl.GestureClassifier(prof),
                             prof["timing"], args, show=False)
        perc.start()
        try:
            _, out = quiet(asyncio.run, gf.main_async(args, perc, False))
        finally:
            perc.stop()
        self.assertIn("NOT recognized", out)
        self.assertNotIn("TAKEOFF", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
