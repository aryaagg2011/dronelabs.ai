import cv2
import depthai as dai

with dai.Pipeline() as pipeline:
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        q = cam.requestOutput((640, 400)).createOutputQueue()
        pipeline.start()
        while pipeline.isRunning():
                frame = q.get().getCvFrame()
                cv2.imshow("rgb", frame)
                if cv2.waitKey(1) == ord("q"):
                        break