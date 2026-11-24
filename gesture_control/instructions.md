# gesture_control

Fly the X500 with hand signals: teach your own signals once, then reuse them every flight.

| File                    | What it does                                                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `gesture_setup.py`      | Teach WAKE, LEFT, RIGHT, FORWARD, BACKWARD, LAND and "relaxed hands"; checks them, live-tests them, saves `gesture_profile.json` |
| `gesture_flight.py`     | Recheck on the ground → take off → hover until WAKE → hold-to-move → LAND. Simulated unless `--live`                             |
| `gesture_lib.py`        | Shared code: camera pipeline, MediaPipe hands, matching, debouncing, person tracking, profile file                               |
| `test_gesture_logic.py` | Checks the logic with a fake camera and fake hands (no hardware). Run after any change                                           |

## Install (Jetson, once)

```
python3 -m pip install -U depthai opencv-python numpy mediapipe --user
python3 -m pip install "mavsdk<4" --user
```

First runs download the person model (Luxonis) and the hand model (~8 MB, Google), so the Jetson needs internet once.

Copy all four `.py` files into the same folder as `remote_view.py` (e.g. `~/oak`).

## Order of testing

```
python3 test_gesture_logic.py                       # 1. logic check, no hardware (~1 min)
python3 remote_view.py gesture_setup.py --name <userName> # 2. teach your signals (stand 2-3 m away)
python3 remote_view.py gesture_flight.py            # 3. simulated flight: watch what it WOULD do
python3 remote_view.py gesture_flight.py --live     # 4. props OFF, outdoors, GPS lock
python3 remote_view.py gesture_flight.py --live --fly  # 5. real flight, RC in hand
```

Without `remote_view.py` the setup needs a monitor; the flight script can run with `--no-display`.

Re-teach only some signals: `gesture_setup.py --redo land wake`. Re-test a saved profile: `gesture_setup.py --test-only`.

## Directions

The drone takes off **facing you**. Default `--frame operator` means the signals are _your_ directions:
LEFT = to your left, RIGHT = to your right, FORWARD = away from you, BACKWARD = toward you.
`--frame drone` uses the drone's own directions instead. The startup log prints what each signal does.

Only "toward you" is seen by the depth camera. Sideways and away are blind and use `--blind-speed` (0.3 m/s).

## Safety defaults (all flags)

Hover is the default whenever anything is unclear. Toward you is refused closer than 2.0 m (`--min-operator-dist`) or when depth sees something within 1.2 m. Each move is limited to 4 s (`--max-move`). There is a 4 m soft fence from takeoff (`--fence`). Losing the operator means hover, then landing after 8 s (`--lost-land`). No WAKE within 60 s of takeoff means landing. A camera stall means hover, then landing after 3 s. Pressing q or Ctrl+C lands.
Set PX4's own geofence and failsafes too. The RC transmitter is the real safety net.

## Privacy

`gesture_profile.json` holds numbers describing hand shapes, no pictures. Add it to `.gitignore` along with `enrolled_person.npz` and `target.png`.
