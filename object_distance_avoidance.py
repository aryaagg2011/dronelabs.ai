import cv2
import numpy as np
import depthai as dai

STOP_DIST_MM = 1500      # closer than this -> STOP
WARN_DIST_MM = 3000      # closer than this -> WARNING
MIN_VALID_FRACTION = 0.10  # need at least 10% valid depth pixels to trust the reading
FRAMES_TO_CONFIRM = 3    # state must hold this many frames before it changes

# The corridor: the center part of the image, as fractions of width/height
X_MIN, X_MAX = 0.30, 0.70
Y_MIN, Y_MAX = 0.25, 0.75

with dai.Pipeline() as pipeline:
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    stereo = pipeline.create(dai.node.StereoDepth)
    monoLeft.requestOutput((640, 480)).link(stereo.left)
    monoRight.requestOutput((640, 480)).link(stereo.right)
    stereo.setLeftRightCheck(True)  # discard unreliable matches

    depthQ = stereo.depth.createOutputQueue()
    pipeline.start()

    state = "UNKNOWN"
    pending, pending_count = None, 0

    while pipeline.isRunning():
        depth = depthQ.get().getFrame()  # uint16 array, millimeters, 0 = no data
        h, w = depth.shape
        x1, x2 = int(w * X_MIN), int(w * X_MAX)
        y1, y2 = int(h * Y_MIN), int(h * Y_MAX)
        roi = depth[y1:y2, x1:x2]
        valid = roi[roi > 0]

        # Decide what this frame says
        if valid.size < MIN_VALID_FRACTION * roi.size:
            frame_state, dist = "STOP", None   # can't see anything: treat as unsafe
        else:
            dist = float(np.percentile(valid, 10))  # near-minimum, but ignores single noisy pixels
            if dist < STOP_DIST_MM:
                frame_state = "STOP"
            elif dist < WARN_DIST_MM:
                frame_state = "WARNING"
            else:
                frame_state = "CLEAR"

        # Only change state after several consistent frames (stops flickering)
        if frame_state == state:
            pending, pending_count = None, 0
        elif frame_state == pending:
            pending_count += 1
            if pending_count >= FRAMES_TO_CONFIRM:
                state = frame_state
                pending, pending_count = None, 0
                if dist is None:
                    print(f"{state}: no valid depth in the corridor")
                else:
                    print(f"{state}: nearest obstacle ~{dist/1000:.2f} m")
        else:
            pending, pending_count = frame_state, 1

        # Visualization
        shown = np.clip(depth, 0, 5000).astype(np.float32) / 5000 * 255
        vis = cv2.applyColorMap(shown.astype(np.uint8), cv2.COLORMAP_JET)
        vis[depth == 0] = 0
        color = {"CLEAR": (0, 255, 0), "WARNING": (0, 255, 255),
                 "STOP": (0, 0, 255), "UNKNOWN": (255, 255, 255)}[state]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = state if dist is None else f"{state}  {dist/1000:.2f} m"
        cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Obstacle check", vis)
        if cv2.waitKey(1) == ord("q"):
            break