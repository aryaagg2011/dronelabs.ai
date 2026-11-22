import cv2
import depthai as dai
import blobconverter

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
            "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
            "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Downloads the compiled model once, then reuses the cached copy
nnPath = blobconverter.from_zoo(name="mobilenet-ssd", shaves=6)

pipeline = dai.Pipeline()

# Color camera: 300x300 preview feeds the neural network
camRgb = pipeline.create(dai.node.ColorCamera)
camRgb.setBoardSocket(dai.CameraBoardSocket.CAM_A)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setPreviewSize(300, 300)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

# Mono cameras for stereo depth
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setBoardSocket(dai.CameraBoardSocket.CAM_B)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setBoardSocket(dai.CameraBoardSocket.CAM_C)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

stereo = pipeline.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)  # line depth up with color
stereo.setOutputSize(monoLeft.getResolutionWidth(), monoLeft.getResolutionHeight())
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Detection + depth fusion, running on the camera
sdn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
sdn.setBlobPath(nnPath)
sdn.setConfidenceThreshold(0.5)
sdn.input.setBlocking(False)
sdn.setBoundingBoxScaleFactor(0.5)
sdn.setDepthLowerThreshold(100)    # ignore closer than 10 cm
sdn.setDepthUpperThreshold(8000)   # ignore farther than 8 m
camRgb.preview.link(sdn.input)
stereo.depth.link(sdn.inputDepth)

# Send the frame and detections to the Jetson
xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutRgb.setStreamName("rgb")
sdn.passthrough.link(xoutRgb.input)
xoutDet = pipeline.create(dai.node.XLinkOut)
xoutDet.setStreamName("detections")
sdn.out.link(xoutDet.input)

with dai.Device(pipeline) as device:
    qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
    qDet = device.getOutputQueue("detections", maxSize=4, blocking=False)

    while True:
        frame = qRgb.get().getCvFrame()
        h, w = frame.shape[:2]

        for det in qDet.get().detections:
            x1, y1 = int(det.xmin * w), int(det.ymin * h)
            x2, y2 = int(det.xmax * w), int(det.ymax * h)
            c = det.spatialCoordinates  # millimeters, relative to the camera

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, labelMap[det.label], (x1 + 3, y1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"Z: {c.z/1000:.2f} m", (x1 + 3, y1 + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            print(f"{labelMap[det.label]}: X={c.x/1000:.2f} Y={c.y/1000:.2f} Z={c.z/1000:.2f} m")

        cv2.imshow("Spatial detection", cv2.resize(frame, (600, 600)))
        if cv2.waitKey(1) == ord("q"):
            break