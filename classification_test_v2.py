import cv2
import depthai as dai

with dai.Pipeline() as pipeline:
    # The three cameras: color in the middle, two mono cameras for stereo
    camRgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)

    # Stereo depth, aligned to the color camera so boxes and depth line up
    stereo = pipeline.create(dai.node.StereoDepth)
    monoLeft.requestOutput((640, 480)).link(stereo.left)
    monoRight.requestOutput((640, 480)).link(stereo.right)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)

    # Detection model + depth fusion, running on the camera
    sdn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        camRgb, stereo, "yolov6-nano")
    sdn.setConfidenceThreshold(0.5)
    sdn.input.setBlocking(False)
    sdn.setBoundingBoxScaleFactor(0.5)  # measure depth from the box center
    sdn.setDepthLowerThreshold(100)     # ignore closer than 10 cm
    sdn.setDepthUpperThreshold(8000)    # ignore farther than 8 m
    labels = sdn.getClasses()

    frameQ = sdn.passthrough.createOutputQueue()
    detQ = sdn.out.createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        frame = frameQ.get().getCvFrame()
        h, w = frame.shape[:2]

        for det in detQ.get().detections:
            x1, y1 = int(det.xmin * w), int(det.ymin * h)
            x2, y2 = int(det.xmax * w), int(det.ymax * h)
            c = det.spatialCoordinates  # millimeters, relative to the camera
            name = labels[det.label]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {det.confidence:.0%}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"X:{c.x/1000:.2f} Y:{c.y/1000:.2f} Z:{c.z/1000:.2f} m",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            print(f"{name}: X={c.x/1000:.2f} Y={c.y/1000:.2f} Z={c.z/1000:.2f} m")

        cv2.imshow("Spatial detection", frame)
        if cv2.waitKey(1) == ord("q"):
            break