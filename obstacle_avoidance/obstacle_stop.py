#!/usr/bin/env python3
"""
Obstacle-stop test: OAK-D Lite depth + PX4 offboard control (MAVSDK).

Sequence: arm -> takeoff -> fly forward slowly -> STOP the moment the depth
camera sees something closer than --stop-dist.

TWO MODES
  (no flags)  SIMULATED. Nothing is sent to the drone. The script prints
              every action it WOULD take, while the depth window shows what
              the camera sees. Start here.
  --live      Sends REAL commands to the Pixhawk. PROPELLERS MUST BE OFF.
              Needs a GPS lock (PX4 refuses takeoff/velocity commands
              without a position estimate), so this means testing outdoors.

WHAT "STOP" MEANS
  Default: send zero velocity (hover), hold 2 s, then land. Motors spin down
           when the landing finishes.
  --bench-kill: PROPS-OFF BENCH TESTING ONLY. Cuts the motors instantly.
           In the air this would make the drone fall.

FAIL-SAFE: no depth data, a stalled camera, or a blank/too-close surface all
count as "obstacle". The script only moves when it can positively see clear space.

Press q in the depth window, or Ctrl+C, to stop and land at any time.

Requires DepthAI v3:   python3 -m pip install -U depthai opencv-python numpy mavsdk
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import depthai as dai

# ---------------------------------------------------------------- settings
ROI_WIDTH = 0.40           # share of image width that is checked (centered)
ROI_HEIGHT = 0.40          # share of image height that is checked (centered)
MIN_VALID_FRACTION = 0.10  # fewer valid depth pixels than this => treat as blocked
NEAR_PERCENTILE = 5        # distance estimate = nearest 5% of pixels (ignores speckle)
STALE_AFTER_S = 0.5        # no new depth frame for this long => treat as blocked
DISPLAY_MAX_MM = 5000      # depth range mapped to the color scale in the window
TAKEOFF_WAIT_S = 8.0       # time given for takeoff before moving forward
HOLD_S = 2.0               # time spent hovering after the stop, before landing
CONTROL_HZ = 10            # offboard setpoint rate (PX4 needs more than 2 Hz)
WINDOW = "OAK-D depth  (q = quit)"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------- perception
@dataclass
class Perception:
    distance_m: Optional[float] = None
    blocked: bool = True
    reason: str = "waiting for first depth frame"
    last_frame_time: float = 0.0
    flight_text: str = "STARTING"

    def is_blocked(self) -> bool:
        return self.blocked or self.is_stale()

    def is_stale(self) -> bool:
        return (time.monotonic() - self.last_frame_time) > STALE_AFTER_S

    def describe(self) -> str:
        if self.is_stale():
            return "CAMERA STALLED (no frames)"
        if self.distance_m is None:
            return self.reason
        return f"{self.reason} {self.distance_m:.2f} m"


def analyze_depth(
    depth_mm: np.ndarray, stop_dist_m: float
) -> Tuple[Optional[float], bool, str, Tuple[int, int, int, int]]:
    """Return (nearest_distance_m, blocked, reason, roi_box) for one depth frame.

    depth_mm is a uint16 array in millimetres, 0 = no valid depth.
    """
    h, w = depth_mm.shape
    x0, x1 = int(w * (0.5 - ROI_WIDTH / 2)), int(w * (0.5 + ROI_WIDTH / 2))
    y0, y1 = int(h * (0.5 - ROI_HEIGHT / 2)), int(h * (0.5 + ROI_HEIGHT / 2))
    box = (x0, y0, x1, y1)

    roi = depth_mm[y0:y1, x0:x1]
    valid = roi[roi > 0]
    if valid.size < MIN_VALID_FRACTION * roi.size:
        return None, True, "NO DEPTH DATA (too close or blank surface)", box

    nearest_m = float(np.percentile(valid, NEAR_PERCENTILE)) / 1000.0
    if nearest_m < stop_dist_m:
        return nearest_m, True, "OBSTACLE", box
    return nearest_m, False, "CLEAR", box


def put_text(img, text, org, scale, thickness=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 3)
    cv2.putText(img, text, org, FONT, scale, (255, 255, 255), thickness)


def draw_view(depth_mm: np.ndarray, box, st: Perception) -> np.ndarray:
    scaled = np.clip(depth_mm, 0, DISPLAY_MAX_MM).astype(np.float32) / DISPLAY_MAX_MM
    # near = red, far = blue, invalid = black
    img = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_JET)
    img[depth_mm == 0] = 0

    color = (0, 0, 255) if st.blocked else (0, 170, 0)
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)
    cv2.rectangle(img, (0, 0), (img.shape[1], 60), color, -1)
    banner = ("STOP: " if st.blocked else "") + st.describe()
    put_text(img, banner, (10, 42), 0.9, 2)
    put_text(img, st.flight_text, (10, img.shape[0] - 15), 0.8, 2)
    return img


async def perception_loop(st: Perception, args, stop_event: asyncio.Event, show: bool):
    """Runs the OAK-D stereo pipeline and keeps `st` up to date."""
    try:
        with dai.Pipeline() as pipeline:
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth).build(
                left.requestOutput((640, 400)),
                right.requestOutput((640, 400)),
                presetMode=dai.node.StereoDepth.PresetMode.ROBOTICS,
            )
            depth_q = stereo.depth.createOutputQueue(maxSize=1, blocking=False)
            pipeline.start()
            log("Depth pipeline running")

            last_print = 0.0
            while not stop_event.is_set() and pipeline.isRunning():
                msg = depth_q.tryGet()
                if msg is not None:
                    depth = msg.getFrame()
                    dist, blocked, reason, box = analyze_depth(depth, args.stop_dist)
                    st.distance_m, st.blocked, st.reason = dist, blocked, reason
                    st.last_frame_time = time.monotonic()
                    if show:
                        cv2.imshow(WINDOW, draw_view(depth, box, st))
                    elif time.monotonic() - last_print > 0.5:
                        last_print = time.monotonic()
                        print(f"   depth: {st.describe():<55} | {st.flight_text}", flush=True)
                if show and (cv2.waitKey(1) & 0xFF == ord("q")):
                    log("q pressed - aborting")
                    stop_event.set()
                await asyncio.sleep(0.01)
    except Exception as e:  # camera unplugged, not found, etc.
        log(f"DEPTH ERROR: {e}")  # frames stop -> flight logic treats it as blocked
    finally:
        if show:
            cv2.destroyAllWindows()


# --------------------------------------------------------------- commanders
class SimCommander:
    """Prints what would happen. Sends nothing anywhere."""

    def __init__(self):
        self.armed = False
        self._last_v = None

    async def preflight(self):
        log("[SIM] (simulated flight - nothing is sent to the drone)")

    async def arm_and_takeoff(self, altitude):
        log(f"[SIM] ARM + TAKEOFF to {altitude:.1f} m")
        self.armed = True
        await asyncio.sleep(3.0)

    async def start_offboard(self):
        log("[SIM] OFFBOARD mode started")

    async def set_forward(self, v):
        if v != self._last_v:
            log(f"[SIM] velocity command: forward {v:.1f} m/s")
            self._last_v = v

    async def stop_and_land(self):
        log("[SIM] LAND")
        await asyncio.sleep(1.5)
        log("[SIM] landed, motors stopped")
        self.armed = False

    async def kill(self):
        log("[SIM] *** MOTORS KILLED ***")
        self.armed = False


class MavCommander:
    """Real commands to the Pixhawk through MAVSDK."""

    def __init__(self, address: str):
        from mavsdk import System
        from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

        self.address = address
        self.drone = System()
        self._Vel = VelocityBodyYawspeed
        self._OffboardError = OffboardError
        self.armed = False
        self.in_offboard = False

    async def _wait_connected(self):
        async for s in self.drone.core.connection_state():
            if s.is_connected:
                return

    async def _wait_position(self):
        async for h in self.drone.telemetry.health():
            if h.is_global_position_ok and h.is_home_position_ok:
                return

    async def preflight(self):
        log(f"Connecting to flight controller at {self.address} ...")
        await self.drone.connect(system_address=self.address)
        try:
            await asyncio.wait_for(self._wait_connected(), 20)
        except asyncio.TimeoutError:
            raise RuntimeError("No heartbeat from the Pixhawk (check cable, port, baud).")
        log("Flight controller connected")
        try:
            await asyncio.wait_for(self._wait_position(), 15)
        except asyncio.TimeoutError:
            raise RuntimeError(
                "No GPS position lock. PX4 refuses takeoff and velocity commands "
                "without a position estimate. Test outdoors (props off) and wait for "
                "a fix, or run without --live."
            )
        log("Position lock OK")

    async def arm_and_takeoff(self, altitude):
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

    async def set_forward(self, v):
        await self.drone.offboard.set_velocity_body(self._Vel(v, 0.0, 0.0, 0.0))

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

    async def kill(self):
        log("*** KILL: motors cut ***")
        await self.drone.action.kill()
        self.armed = False
        self.in_offboard = False


# ----------------------------------------------------------- flight sequence
async def flight_sequence(cmd, st: Perception, args, stop_event: asyncio.Event):
    # Do not start until the camera is delivering frames.
    t0 = time.monotonic()
    while st.last_frame_time == 0.0:
        if time.monotonic() - t0 > 15:
            raise RuntimeError("No depth frames from the OAK-D after 15 s - not flying.")
        await asyncio.sleep(0.1)
    log("Depth camera OK")

    await cmd.preflight()

    try:
        st.flight_text = "ARM + TAKEOFF"
        await cmd.arm_and_takeoff(args.altitude)

        if st.is_blocked() or stop_event.is_set():
            log(f"Not moving: {st.describe()}")
            st.flight_text = "NOT MOVING - obstacle/no data at start"
        else:
            await cmd.start_offboard()
            log(f"Flying forward at {args.speed:.1f} m/s, stop distance {args.stop_dist:.2f} m")
            t_start = time.monotonic()
            stopped = False
            while time.monotonic() - t_start < args.max_forward:
                if st.is_blocked() or stop_event.is_set():
                    stopped = True
                    break
                st.flight_text = f"FORWARD {args.speed:.1f} m/s"
                await cmd.set_forward(args.speed)
                await asyncio.sleep(1.0 / CONTROL_HZ)

            await cmd.set_forward(0.0)  # <- the stop command
            if stopped:
                log("!!! STOP !!!  " + ("operator abort" if stop_event.is_set() else st.describe()))
                st.flight_text = "STOPPED - obstacle detected"
            else:
                log("Reached max forward time with no obstacle - stopping")
                st.flight_text = "STOPPED - time limit"

            if args.bench_kill:
                await cmd.kill()
                st.flight_text += " | MOTORS KILLED"
            else:
                for _ in range(int(HOLD_S * CONTROL_HZ)):  # keep streaming zero velocity
                    await cmd.set_forward(0.0)
                    await asyncio.sleep(1.0 / CONTROL_HZ)
    except Exception as e:
        log(f"ERROR: {e}")
        st.flight_text = "ERROR - landing"
    finally:
        if cmd.armed:  # normal path, abort, or error: never leave it flying
            await cmd.stop_and_land()
        st.flight_text += " | DONE"
        log("Sequence finished")
        await asyncio.sleep(1.0)  # let the window show the final state
        stop_event.set()


async def main_async(args):
    st = Perception()
    stop_event = asyncio.Event()
    show = (not args.no_display) and bool(os.environ.get("DISPLAY"))
    if not show:
        log("No display - printing depth status to the terminal instead of a window")

    cmd = MavCommander(args.address) if args.live else SimCommander()
    perception = asyncio.create_task(perception_loop(st, args, stop_event, show))
    try:
        await flight_sequence(cmd, st, args, stop_event)
    except Exception as e:
        log(f"ERROR: {e}")
    finally:
        stop_event.set()
        await perception


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="OAK-D obstacle-stop test for PX4 (propellers OFF).")
    p.add_argument("--live", action="store_true", help="send REAL commands (props off!)")
    p.add_argument("--address", default="serial:///dev/ttyTHS1:921600", help="MAVSDK connection string")
    p.add_argument("--stop-dist", type=float, default=1.2, help="stop if something is closer than this (m)")
    p.add_argument("--speed", type=float, default=0.5, help="forward speed (m/s)")
    p.add_argument("--altitude", type=float, default=1.5, help="takeoff altitude (m)")
    p.add_argument("--max-forward", type=float, default=15.0, help="max seconds of forward motion")
    p.add_argument("--no-display", action="store_true", help="no window; print status in the terminal")
    p.add_argument("--bench-kill", action="store_true", help="PROPS-OFF ONLY: cut motors on stop")
    return p.parse_args(argv)


def main():
    if int(dai.__version__.split(".")[0]) < 3:
        sys.exit("This script needs DepthAI v3. Run: python3 -m pip install -U depthai")

    args = parse_args()
    if args.live:
        print("\n*** LIVE MODE: this will ARM the motors and send flight commands. ***")
        if input("Propellers removed? Type  PROPS OFF  to continue: ").strip() != "PROPS OFF":
            sys.exit("Aborted.")
    elif args.bench_kill:
        log("(--bench-kill only matters together with --live)")

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        log("Ctrl+C - exiting")


if __name__ == "__main__":
    main()