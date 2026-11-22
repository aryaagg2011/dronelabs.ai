import cv2
import numpy as np
import depthai as dai

pipeline = dai.Pipeline()

left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
stereo = pipeline.create(dai.node.StereoDepth)

left.requestFullResolutionOutput().link(stereo.left)
right.requestFullResolutionOutput().link(stereo.right)

stereo.setRectification(True)
stereo.setExtendedDisparity(True)   # better for close range
stereo.setLeftRightCheck(True)      # better edges around objects

dispQ = stereo.disparity.createOutputQueue()
depthQ = stereo.depth.createOutputQueue()

with pipeline:
    pipeline.start()
    maxDisp = stereo.initialConfig.getMaxDisparity()
    while pipeline.isRunning():
        disp = dispQ.get().getFrame()
        depth = depthQ.get().getFrame()   # uint16, millimetres
        h, w = depth.shape
        print(f"center: {depth[h // 2, w // 2] / 1000:.2f} m", end="\r")
        vis = (disp * (255.0 / maxDisp)).astype(np.uint8)
        cv2.imshow("disparity", cv2.applyColorMap(vis, cv2.COLORMAP_JET))
        if cv2.waitKey(1) == ord("q"):
            break