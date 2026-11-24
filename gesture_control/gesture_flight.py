#!/usr/bin/env python3
"""
gesture_flight.py - fly the drone with the hand signals saved by gesture_setup.py.

SEQUENCE
  1. Load your profile (gesture_profile.json). It must be validated by gesture_setup.py.
  2. Connect to the Pixhawk (live modes only) and wait for a GPS position.
  3. RECHECK on the ground: the screen/terminal asks for each signal in turn (WAKE, LEFT,
     RIGHT, FORWARD, BACKWARD, LAND). Show each one until it is confirmed. Lighting and
     clothing change from day to day; if any signal is not recognized within 20 s, it does
     not take off.
  4. Take off and HOVER. Nothing moves until you hold your WAKE signal (1 s). The person
     who wakes it becomes the operator; signals from anyone else are ignored.
  5. HOLD TO MOVE: hold LEFT / RIGHT / FORWARD / BACKWARD and it moves; drop the signal
     and it stops within ~0.3 s. Hold LAND (1.5 s) and it lands.

MODES
  (no flags)      SIMULATED. Nothing is sent to the drone; it prints what it WOULD do.
                  Start here. Works indoors, no Pixhawk needed.
  --live          Real commands, PROPELLERS OFF (bench test). Needs a GPS lock, so outdoors.
  --live --fly    Real flight with propellers on. Asks you to confirm a safety checklist.

SAFETY BEHAVIOUR
  * Default is hover. No signal, an unclear signal, two hands showing different signals,
    or the operator not visible -> zero velocity.
  * Operator gone for more than ~1 s -> forgotten; hovers and waits for WAKE again.
    Not re-woken within --lost-land seconds (8) -> lands.
  * No WAKE within --wake-timeout seconds (60) after takeoff -> lands.
  * Camera stalled -> hovers; stalled more than --stale-land seconds (3) -> lands.
  * Moving toward the camera's side (toward you) is refused when the depth camera sees
    something closer than --stop-dist (1.2 m), when you are closer than --min-operator-dist
    (2.0 m), or when your distance is unknown.
  * Moving away from the camera's side and sideways is BLIND (the camera only looks forward),
    so it uses the lower --blind-speed (0.3 m/s). Fly in a large open area.
  * A signal held longer than --max-move seconds (4) stops; release and signal again.
  * Soft fence: refuses to move more than --fence metres (4) from the takeoff point.
    Set PX4's own geofence too, and keep the RC transmitter in your hand.
  * q in the window (or Q in the browser viewer), or Ctrl+C: stop and land.

DIRECTIONS (--frame)
  The drone takes off facing you (point the camera at where you will stand).
  --frame operator (default): directions are YOURS.  LEFT = to your left, RIGHT = to your
        right, FORWARD = away from you, BACKWARD = toward you.
  --frame drone: directions are the DRONE's.  FORWARD = the way the camera looks (toward you),
        LEFT = the drone's left (your right).
  The startup log prints exactly what each signal will do.

USAGE
    python3 gesture_flight.py                          # simulated
    python3 remote_view.py gesture_flight.py           # simulated, watch in the browser
    python3 gesture_flight.py --live                   # props OFF, real commands
    python3 gesture_flight.py --live --fly             # real flight

Requires: python3 -m pip install -U depthai opencv-python numpy mediapipe "mavsdk<4"
"""

import argparse
import asyncio
import math
import sys
import threading
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import gesture_lib as gl
from gesture_lib import log

WINDOW = "Gesture flight  (q = stop and land)"
CONTROL_HZ = 10            # offboard setpoint rate (PX4 needs more than 2 Hz)
STALE_S = 0.5              # no camera frame for this long -> hover
LOST_HOVER_S = 0.5         # operator box not seen for this long -> hover
TAKEOFF_WAIT_S = 8.0
FENCE_LOOKAHEAD_S = 0.5
YAW_GAIN = 0.5             # deg/s of turn per degree the operator is off-centre (--yaw-track)
YAW_MAX = 15.0
YAW_DEADBAND = 5.0
MAX_CANDIDATES = 2         # how many nearby people are checked for WAKE each frame

# body-frame direction (forward, right) for each signal
DIRECTIONS = {
    "drone": {gl.FORWARD: (1, 0), gl.BACKWARD: (-1, 0), gl.LEFT: (0, -1), gl.RIGHT: (0, 1)},
    # facing each other: your left is the drone's right, "away from you" is the drone's backward
    "operator": {gl.FORWARD: (-1, 0), gl.BACKWARD: (1, 0), gl.LEFT: (0, 1), gl.RIGHT: (0, -1)},
}


def describe_direction(vec) -> str:
    return {(1, 0): "toward the camera's view (toward you)", (-1, 0): "away from you (blind)",
            (0, 1): "to the drone's right = YOUR left (blind)",
            (0, -1): "to the drone's left = YOUR right (blind)"}[tuple(vec)]


# ------------------------------------------------------------------ perception
@dataclass
class Snapshot:
    frame_time: float = 0.0
    fps: float = 0.0
    people: int = 0
    operator_id: Optional[int] = None
    operator_dist_m: Optional[float] = None
    operator_bearing_deg: Optional[float] = None
    operator_seen: float = 0.0
    lock_time: float = 0.0
    lost_time: float = 0.0
    label: str = gl.NONE
    active: Optional[str] = None
    active_since: float = 0.0
    land_events: int = 0
    held: frozenset = frozenset()
    depth_m: Optional[float] = None
    depth_blocked: bool = True
    depth_reason: str = "waiting"
    error: str = ""


class Perception:
    """Camera + hands + operator tracking, in its own thread so slow hand detection never
    delays the flight commands. The flight code only reads snapshot()."""

    def __init__(self, source, detector, clf: gl.GestureClassifier, timing: dict, args, show: bool,
                 min_palm_px: float = 10.0):
        self.source, self.detector, self.clf = source, detector, clf
        self.timing, self.args, self.show = timing, args, show
        self.min_palm_px = min_palm_px
        self.tracker = gl.PersonTracker(max_age=1.0)
        self.op_deb = gl.GestureDebouncer(timing)
        self.cand_debs: Dict[int, gl.GestureDebouncer] = {}
        self.allow_lock = False          # set by the flight code
        self.banner: List[str] = []      # set by the flight code
        self.banner_color = (60, 60, 60)
        self._st = Snapshot()            # owned by the perception thread
        self._pub = Snapshot()           # published copy
        self._view = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._t_prev = None

    # -- called from the flight code
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3)

    def snapshot(self) -> Snapshot:
        with self._lock:
            return self._pub

    def latest_view(self):
        with self._lock:
            return self._view

    def set_banner(self, lines, color=(60, 60, 60)):
        self.banner, self.banner_color = list(lines), color

    # -- the thread
    def _run(self):
        while not self._stop.is_set():
            try:
                frame = self.source.read(timeout=0.5)
            except Exception as e:      # camera unplugged etc.: frames stop -> flight hovers/lands
                self._fail(f"camera error: {e}")
                return
            if frame is None:
                if not self.source.is_running():
                    self._fail("camera stopped")
                    return
                continue
            try:
                self.process(frame)
            except Exception as e:
                self._fail(f"perception error: {e}")
                return

    def _fail(self, msg: str):
        log(f"PERCEPTION STOPPED: {msg}")
        with self._lock:
            self._pub = replace(self._pub, error=msg)

    def process(self, frame: gl.Frame) -> None:
        st, now = self._st, frame.t
        tracked = self.tracker.update(frame.people, now)
        if self._t_prev is not None:
            st.fps = 0.9 * st.fps + 0.1 / max(now - self._t_prev, 1e-3)
        self._t_prev = now
        st.frame_time, st.people = now, len(tracked)
        st.depth_m, st.depth_blocked, st.depth_reason = gl.analyze_depth(frame.depth_mm, self.args.stop_dist)

        if st.operator_id is not None and not self.tracker.alive(st.operator_id):
            log(f"OPERATOR #{st.operator_id} LOST")
            st.operator_id, st.lost_time = None, now
            st.operator_dist_m = st.operator_bearing_deg = None
            self.op_deb.reset()

        results: Dict[int, gl.HandsResult] = {}
        if st.operator_id is not None:
            p = tracked.get(st.operator_id)
            label = gl.NONE
            if p is not None:
                res = gl.analyze_hands(frame, p, self.detector, self.clf, self.min_palm_px)
                results[st.operator_id] = res
                label = res.label
                st.operator_seen = now
                st.operator_dist_m, st.operator_bearing_deg = p.distance_m, p.bearing_deg
            if gl.LAND in self.op_deb.update(now, label):
                st.land_events += 1
                log("LAND signal held")
            st.label, st.active, st.active_since = label, self.op_deb.active, self.op_deb.active_since
            st.held = frozenset(g for g in gl.GESTURES if self.op_deb.held(g, self.timing["recheck_hold_s"]))
        else:
            st.active, st.label = None, gl.NONE
            cands = self._candidates(tracked)
            for tid, p in cands:
                res = gl.analyze_hands(frame, p, self.detector, self.clf, self.min_palm_px)
                results[tid] = res
                deb = self.cand_debs.setdefault(tid, gl.GestureDebouncer(self.timing))
                events = deb.update(now, res.label)
                if self.allow_lock and gl.WAKE in events and st.operator_id is None:
                    st.operator_id, st.lock_time, st.operator_seen = tid, now, now
                    st.operator_dist_m, st.operator_bearing_deg = p.distance_m, p.bearing_deg
                    self.op_deb.reset()
                    d = p.distance_m
                    log(f"WAKE: operator locked (person #{tid}" + (f", {d:.1f} m)" if d else ")"))
            if cands:
                first_tid = cands[0][0]
                st.label = results[first_tid].label
                deb = self.cand_debs[first_tid]
                st.held = frozenset(g for g in gl.GESTURES if deb.held(g, self.timing["recheck_hold_s"]))
            else:
                st.held = frozenset()
            for tid in [t for t in self.cand_debs if not self.tracker.alive(t)]:
                del self.cand_debs[tid]
            if st.operator_id is not None:
                self.cand_debs.clear()

        view = self._draw(frame, tracked, results) if self.show else None
        with self._lock:
            self._pub = replace(st)
            if view is not None:
                self._view = view

    def _candidates(self, tracked: Dict[int, gl.Person]) -> List[Tuple[int, gl.Person]]:
        """Nearest people within --max-operator-dist (people with no distance reading come last)."""
        items = []
        for tid, p in tracked.items():
            d = p.distance_m
            if d is not None and d > self.args.max_operator_dist:
                continue
            items.append((d if d is not None else 1e9, -p.area, tid, p))
        items.sort(key=lambda x: (x[0], x[1]))
        return [(tid, p) for _, _, tid, p in items[:MAX_CANDIDATES]]

    def _draw(self, frame, tracked, results):
        st = self._st
        if st.operator_id is not None:
            d = f"{st.operator_dist_m:.1f} m" if st.operator_dist_m else "? m"
            info = f"operator #{st.operator_id} {d}   sees: {st.label.upper()}   " \
                   f"moving: {(st.active or 'no').upper()}"
        else:
            info = f"no operator   sees: {st.label.upper()}   {st.people} people"
        depth = f"depth {st.depth_m:.1f} m" if st.depth_m else f"depth: {st.depth_reason}"
        return gl.draw_view(frame.bgr, tracked, st.operator_id, results,
                            (self.banner or ["gesture flight"]) + [info], self.banner_color,
                            [f"{depth}   {st.fps:.0f} fps"])



# ------------------------------------------------------------------ commanders
class SimCommander:
    """Prints what would happen and keeps a pretend position. Sends nothing anywhere."""

    def __init__(self):
        self.armed = False
        self._last_v = None
        self._pos = np.zeros(2)
        self._heading = 0.0
        self._t = None

    async def preflight(self):
        log("[SIM] simulated flight - nothing is sent to the drone")

    async def arm_and_takeoff(self, altitude):
        log(f"[SIM] ARM + TAKEOFF to {altitude:.1f} m")
        self.armed = True
        await asyncio.sleep(2.0)

    async def start_offboard(self):
        log("[SIM] OFFBOARD mode started")
        self._t = time.monotonic()

    async def set_velocity(self, fwd, right, yaw):
        now = time.monotonic()
        if self._t is not None:
            dt = min(now - self._t, 0.5)
            h = math.radians(self._heading)
            self._pos += dt * np.array([fwd * math.cos(h) - right * math.sin(h),
                                        fwd * math.sin(h) + right * math.cos(h)])
            self._heading = (self._heading + yaw * dt) % 360
        self._t = now
        v = (round(fwd, 2), round(right, 2), round(yaw, 1))
        if v != self._last_v:
            log(f"[SIM] velocity: forward {fwd:+.2f} m/s, right {right:+.2f} m/s, yaw {yaw:+.1f} deg/s"
                f"   (position N {self._pos[0]:+.1f} E {self._pos[1]:+.1f} m)")
            self._last_v = v

    def position_ne(self):
        return float(self._pos[0]), float(self._pos[1])

    def heading_deg(self):
        return self._heading

    async def stop_and_land(self):
        log("[SIM] LAND")
        await asyncio.sleep(1.0)
        log("[SIM] landed, motors stopped")
        self.armed = False

    async def close(self):
        pass


class MavCommander:
    """Real commands to the Pixhawk through MAVSDK (mavsdk < 4)."""

    def __init__(self, address: str):
        from mavsdk import System
        from mavsdk.offboard import OffboardError, VelocityBodyYawspeed
        self.address = address
        self.drone = System()
        self._Vel = VelocityBodyYawspeed
        self._OffboardError = OffboardError
        self.armed = False
        self.in_offboard = False
        self._ned = None          # latest (north, east) from PX4
        self._origin = None       # takeoff point
        self._heading = None
        self._tasks: List[asyncio.Task] = []

    async def _wait_connected(self):
        async for s in self.drone.core.connection_state():
            if s.is_connected:
                return

    async def _wait_position(self):
        async for h in self.drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok and h.is_local_position_ok:
                return

    async def _watch_position(self):
        async for pv in self.drone.telemetry.position_velocity_ned():
            self._ned = (pv.position.north_m, pv.position.east_m)

    async def _watch_heading(self):
        async for h in self.drone.telemetry.heading():
            self._heading = h.heading_deg

    async def preflight(self):
        log(f"Connecting to flight controller at {self.address} ...")
        await self.drone.connect(system_address=self.address)
        try:
            await asyncio.wait_for(self._wait_connected(), 20)
        except asyncio.TimeoutError:
            raise RuntimeError("No heartbeat from the Pixhawk (check cable, port, baud).")
        log("Flight controller connected")
        try:
            await asyncio.wait_for(self._wait_position(), 30)
        except asyncio.TimeoutError:
            raise RuntimeError("No GPS position lock. PX4 refuses takeoff and velocity commands without "
                               "a position estimate. Test outdoors and wait for a fix, or run without --live.")
        log("Position lock OK")
        self._tasks = [asyncio.create_task(self._watch_position()),
                       asyncio.create_task(self._watch_heading())]
        for _ in range(50):
            if self._ned is not None and self._heading is not None:
                break
            await asyncio.sleep(0.1)
        if self._ned is None or self._heading is None:
            raise RuntimeError("No position/heading telemetry from PX4.")

    async def arm_and_takeoff(self, altitude):
        self._origin = self._ned
        await self.drone.action.set_takeoff_altitude(altitude)
        log("ARM")
        await self.drone.action.arm()
        self.armed = True
        log(f"TAKEOFF to {altitude:.1f} m")
        await self.drone.action.takeoff()
        await asyncio.sleep(TAKEOFF_WAIT_S)

    async def start_offboard(self):
        await self.drone.offboard.set_velocity_body(self._Vel(0.0, 0.0, 0.0, 0.0))
        try:
            await self.drone.offboard.start()
        except self._OffboardError as e:
            raise RuntimeError(f"PX4 refused offboard mode: {e}")
        self.in_offboard = True
        log("OFFBOARD mode started")

    async def set_velocity(self, fwd, right, yaw):
        await self.drone.offboard.set_velocity_body(self._Vel(fwd, right, 0.0, yaw))

    def position_ne(self):
        if self._ned is None or self._origin is None:
            return None
        return self._ned[0] - self._origin[0], self._ned[1] - self._origin[1]

    def heading_deg(self):
        return self._heading

    async def stop_and_land(self):
        if self.in_offboard:
            try:
                await self.drone.offboard.set_velocity_body(self._Vel(0.0, 0.0, 0.0, 0.0))
                await self.drone.offboard.stop()
            except Exception as e:
                log(f"(offboard stop: {e})")
            self.in_offboard = False
        log("LAND")
        await self.drone.action.land()
        try:
            await asyncio.wait_for(self._wait_landed(), 40)
            log("Landed")
            self.armed = False
        except asyncio.TimeoutError:
            log("Still reports in-air after 40 s (common with props off). "
                "Disarm from QGroundControl or the RC transmitter.")

    async def _wait_landed(self):
        async for in_air in self.drone.telemetry.in_air():
            if not in_air:
                return

    async def close(self):
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass


# ------------------------------------------------------------------ decisions
def velocity_for(active: str, frame: str, speed: float, blind_speed: float) -> Tuple[float, float]:
    f, r = DIRECTIONS[frame][active]
    if f > 0:
        return f * speed, 0.0          # toward the camera's view: checked by depth
    return f * blind_speed, r * blind_speed


def blocked_reason(s: Snapshot, fwd: float, right: float, now: float, cmd, args) -> Optional[str]:
    if now - s.active_since > args.max_move:
        return f"held more than {args.max_move:.0f} s - release and signal again"
    if fwd > 0:
        if s.depth_blocked:
            return f"depth camera: {s.depth_reason}" + (f" {s.depth_m:.1f} m" if s.depth_m else "")
        if s.operator_dist_m is None:
            return "your distance is unknown"
        if s.operator_dist_m < args.min_operator_dist:
            return f"too close to you ({s.operator_dist_m:.1f} m < {args.min_operator_dist:.1f} m)"
    pos, heading = cmd.position_ne(), cmd.heading_deg()
    if pos is None or heading is None:
        return "no position estimate"
    h = math.radians(heading)
    n, e = pos
    dn = fwd * math.cos(h) - right * math.sin(h)
    de = fwd * math.sin(h) + right * math.cos(h)
    now_d = math.hypot(n, e)
    next_d = math.hypot(n + dn * FENCE_LOOKAHEAD_S, e + de * FENCE_LOOKAHEAD_S)
    if next_d > args.fence and next_d >= now_d:
        return f"fence ({now_d:.1f} m from takeoff, limit {args.fence:.1f} m)"
    return None


def yaw_for(s: Snapshot) -> float:
    b = s.operator_bearing_deg
    if b is None or abs(b) < YAW_DEADBAND:
        return 0.0
    return max(-YAW_MAX, min(YAW_MAX, YAW_GAIN * b))


class StatusLog:
    """Print a status line when it changes (and repeat slow-changing ones every few seconds)."""

    def __init__(self, perc: Perception):
        self.perc, self.key, self.last = perc, None, 0.0

    def __call__(self, key: str, text: str, color=(60, 60, 60), repeat_s: float = 3.0):
        now = time.monotonic()
        if key != self.key or now - self.last > repeat_s:
            log(text)
            self.key, self.last = key, now
        self.perc.set_banner([text], color)


async def recheck(perc: Perception, args, abort: asyncio.Event) -> bool:
    perc.allow_lock = False
    log("RECHECK: show each signal until it is confirmed (stand where you will fly from)")
    for i, g in enumerate(gl.GESTURES, 1):
        prompt = f"RECHECK {i}/{len(gl.GESTURES)}: show {g.upper()}"
        log(prompt)
        t0 = time.monotonic()
        while True:
            if abort.is_set():
                return False
            s = perc.snapshot()
            if s.error:
                log(f"Camera problem during recheck: {s.error}")
                return False
            if g in s.held:
                log(f"  {g.upper()} OK")
                break
            left = args.recheck_timeout - (time.monotonic() - t0)
            if left <= 0:
                log(f"  {g.upper()} NOT recognized within {args.recheck_timeout:.0f} s - not flying. "
                    "Check the lighting, stand at your usual distance, or re-run gesture_setup.py.")
                return False
            perc.set_banner([prompt, f"{left:.0f} s left   (reading: {s.label.upper()})"], (120, 90, 0))
            await asyncio.sleep(0.05)
        perc.set_banner([f"{g.upper()} OK"], (0, 140, 0))
        await asyncio.sleep(0.7)
    log("RECHECK passed: all signals recognized")
    return True


async def control_loop(cmd, perc: Perception, args, abort: asyncio.Event) -> str:
    """Runs while airborne. Returns why it ended (the caller then lands)."""
    perc.allow_lock = True
    status = StatusLog(perc)
    t_start = time.monotonic()
    ever_locked = False
    seen_land = perc.snapshot().land_events
    stale_since: Optional[float] = None
    dt = 1.0 / CONTROL_HZ
    while True:
        now = time.monotonic()
        if abort.is_set():
            return "operator abort (q / Ctrl+C)"
        s = perc.snapshot()
        fwd = right = yaw = 0.0

        if s.error or now - s.frame_time > STALE_S:
            stale_since = stale_since or now
            status("stale", "CAMERA STALLED - hovering", (0, 0, 160))
            if now - stale_since > args.stale_land:
                return "camera stalled"
        else:
            stale_since = None
            if s.land_events > seen_land:
                return "LAND signal"
            if s.operator_id is None:
                if ever_locked:
                    lost_for = now - s.lost_time
                    status("lost", f"OPERATOR LOST - hovering; show WAKE "
                                   f"(landing in {max(0.0, args.lost_land - lost_for):.0f} s)", (0, 0, 160))
                    if lost_for > args.lost_land:
                        return "operator lost"
                else:
                    waited = now - t_start
                    status("wait", f"Hovering - hold WAKE to take control "
                                   f"(landing in {max(0.0, args.wake_timeout - waited):.0f} s)", (120, 90, 0),
                           repeat_s=10)
                    if waited > args.wake_timeout:
                        return "no WAKE signal"
            else:
                ever_locked = True
                if now - s.operator_seen > LOST_HOVER_S:
                    status("unseen", "operator not visible - hovering", (0, 0, 160))
                elif s.active:
                    vf, vr = velocity_for(s.active, args.frame, args.speed, args.blind_speed)
                    why = blocked_reason(s, vf, vr, now, cmd, args)
                    if why:
                        status(f"blocked-{s.active}-{why[:12]}", f"{s.active.upper()} refused: {why}",
                               (0, 120, 200))
                    else:
                        fwd, right = vf, vr
                        status(f"move-{s.active}", f"MOVING {s.active.upper()}", (0, 140, 0))
                else:
                    status("hover", "operator locked - hovering (no signal)", (0, 100, 0))
                if args.yaw_track and now - s.operator_seen <= LOST_HOVER_S:
                    yaw = yaw_for(s)
        await cmd.set_velocity(fwd, right, yaw)
        await asyncio.sleep(dt)


async def flight_sequence(cmd, perc: Perception, args, abort: asyncio.Event):
    t0 = time.monotonic()
    while perc.snapshot().frame_time == 0.0:
        if perc.snapshot().error or time.monotonic() - t0 > 20:
            raise RuntimeError("No pictures from the camera - not flying.")
        await asyncio.sleep(0.1)
    log("Camera OK")

    await cmd.preflight()
    if args.skip_recheck:
        log("Recheck skipped (--skip-recheck)")
    elif not await recheck(perc, args, abort):
        return
    for n in (3, 2, 1):
        if abort.is_set():
            return
        log(f"Taking off in {n}...")
        perc.set_banner([f"TAKING OFF IN {n}"], (0, 0, 160))
        await asyncio.sleep(1.0)

    try:
        await cmd.arm_and_takeoff(args.altitude)
        await cmd.start_offboard()
        reason = await control_loop(cmd, perc, args, abort)
        log(f"Ending flight: {reason}")
        perc.set_banner([f"LANDING: {reason}"], (0, 0, 160))
    except Exception as e:
        log(f"ERROR: {e} - landing")
    finally:
        if cmd.armed:          # normal end, abort or error: never leave it flying
            await cmd.stop_and_land()
        log("Sequence finished")


async def ui_loop(perc: Perception, abort: asyncio.Event, done: asyncio.Event, show: bool):
    while not done.is_set():
        if show:
            view = perc.latest_view()
            if view is not None:
                cv2.imshow(WINDOW, view)
            if cv2.waitKey(1) & 0xFF == ord("q") and not abort.is_set():
                log("q pressed - stopping")
                abort.set()
        await asyncio.sleep(0.05)


async def main_async(args, perc: Perception, show: bool):
    abort, done = asyncio.Event(), asyncio.Event()
    cmd = MavCommander(args.address) if args.live else SimCommander()
    ui = asyncio.create_task(ui_loop(perc, abort, done, show))
    try:
        await flight_sequence(cmd, perc, args, abort)
    except asyncio.CancelledError:
        if cmd.armed:
            await cmd.stop_and_land()
        raise
    except Exception as e:
        log(f"ERROR: {e}")
    finally:
        await cmd.close()
        await asyncio.sleep(1.0)       # let the window show the final state
        done.set()
        await ui


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fly with your saved hand signals (simulated by default).")
    p.add_argument("--profile", default="gesture_profile.json")
    p.add_argument("--allow-unvalidated", action="store_true", help="accept a profile that failed setup tests")
    p.add_argument("--live", action="store_true", help="send REAL commands (props off unless --fly)")
    p.add_argument("--fly", action="store_true", help="with --live: real flight, propellers on")
    p.add_argument("--address", default="serial:///dev/ttyTHS1:921600", help="MAVSDK connection string")
    p.add_argument("--frame", choices=["operator", "drone"], default="operator",
                   help="whose left/right/forward the signals mean (default: yours)")
    p.add_argument("--altitude", type=float, default=1.5, help="takeoff altitude (m)")
    p.add_argument("--speed", type=float, default=0.5, help="speed toward the camera's view (m/s)")
    p.add_argument("--blind-speed", type=float, default=0.3, help="speed sideways/away (camera can't see) (m/s)")
    p.add_argument("--stop-dist", type=float, default=1.2, help="refuse to move toward anything closer (m)")
    p.add_argument("--min-operator-dist", type=float, default=2.0, help="never come closer to you than this (m)")
    p.add_argument("--max-operator-dist", type=float, default=4.0,
                   help="only people closer than this can WAKE it (m)")
    p.add_argument("--max-move", type=float, default=4.0, help="longest continuous move per signal (s)")
    p.add_argument("--fence", type=float, default=4.0, help="max distance from the takeoff point (m)")
    p.add_argument("--wake-timeout", type=float, default=60.0, help="land if not woken within this (s)")
    p.add_argument("--lost-land", type=float, default=8.0, help="land if the operator is gone this long (s)")
    p.add_argument("--stale-land", type=float, default=3.0, help="land if the camera stalls this long (s)")
    p.add_argument("--recheck-timeout", type=float, default=20.0, help="time allowed per signal in the recheck (s)")
    p.add_argument("--skip-recheck", action="store_true", help="skip the pre-flight recheck (simulation only)")
    p.add_argument("--yaw-track", action="store_true", help="turn slowly to keep you centred in the picture")
    p.add_argument("--no-display", action="store_true", help="no window; print status in the terminal")
    cam = p.add_argument_group("camera")
    cam.add_argument("--fps", type=float, default=15)
    cam.add_argument("--frame-width", type=int, default=None,
                     help="picture width for hands (default: the width used during setup)")
    cam.add_argument("--model", default="yolov6-nano")
    cam.add_argument("--conf", type=float, default=0.5)
    cam.add_argument("--shaves", type=int, default=None)
    args = p.parse_args(argv)
    if args.fly and not args.live:
        p.error("--fly only makes sense together with --live")
    if args.skip_recheck and args.live:
        p.error("--skip-recheck is only allowed in simulation")
    for name in ("speed", "blind_speed"):
        if not 0 < getattr(args, name) <= 1.5:
            p.error(f"--{name.replace('_', '-')} must be between 0 and 1.5 m/s")
    return args


def confirm_live(args) -> None:
    if not args.live:
        return
    if not args.fly:
        print("\n*** LIVE MODE (bench): this will ARM the motors and send flight commands. ***")
        if input("Propellers removed? Type  PROPS OFF  to continue: ").strip() != "PROPS OFF":
            sys.exit("Aborted.")
        return
    print("\n*** REAL FLIGHT with propellers. Before continuing, check: ***\n"
          "  - large open area, no people other than you nearby\n"
          "  - PX4 geofence and failsafes set; RC transmitter in hand and you know how to take over\n"
          "  - the camera faces where you will stand, 2-3 m away\n"
          "  - this profile passed its tests today in similar light")
    if input("Type  FLY  to continue: ").strip() != "FLY":
        sys.exit("Aborted.")


def main():
    args = parse_args()
    try:
        profile = gl.load_profile(args.profile)
    except FileNotFoundError:
        sys.exit(f"No profile ({args.profile}). Run gesture_setup.py first.")
    except ValueError as e:
        sys.exit(str(e))
    if not profile.get("validated") and not args.allow_unvalidated:
        sys.exit(f"{args.profile} did not pass all setup tests. Re-run gesture_setup.py, "
                 "or add --allow-unvalidated (not recommended outside simulation).")
    meta = profile.get("meta", {})
    if args.frame_width is None:
        args.frame_width = int(meta.get("frame_width", 1280))

    log(f"Profile '{profile.get('name')}' ({profile.get('created')}), "
        f"{'validated' if profile.get('validated') else 'NOT validated'}")
    log(f"Signals (--frame {args.frame}):")
    for g in gl.MOVES:
        vec = DIRECTIONS[args.frame][g]
        spd = args.speed if vec[0] > 0 else args.blind_speed
        log(f"   {g.upper():9s} -> {describe_direction(vec)}  at {spd:.1f} m/s")
    log(f"   {'LAND':9s} -> hold {profile['timing']['land_hold_s']:.1f} s to land;  "
        f"WAKE -> hold {profile['timing']['wake_hold_s']:.1f} s to take control")

    confirm_live(args)
    show = gl.display_available(args.no_display)
    if not show:
        log("No display - printing status in the terminal (use remote_view.py to see the picture)")

    clf = gl.GestureClassifier(profile)
    detector = gl.HandDetector(num_hands=2)
    rig = gl.CameraRig(args.model, args.fps, args.frame_width, args.conf, args.shaves)
    perc = None
    try:
        rig.open()
        perc = Perception(rig, detector, clf, profile["timing"], args, show,
                          float(meta.get("min_palm_px", 10.0)))
        perc.start()
        asyncio.run(main_async(args, perc, show))
    except KeyboardInterrupt:
        log("Ctrl+C - exiting")
    finally:
        if perc is not None:
            perc.stop()
        if show:
            cv2.destroyAllWindows()
        rig.close()
        detector.close()


if __name__ == "__main__":
    main()
