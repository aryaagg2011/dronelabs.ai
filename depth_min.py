import time
import depthai as dai

RUN_SECONDS = 20

print("depthai version:", dai.__version__)
with dai.Pipeline() as pipeline:
    device = pipeline.getDefaultDevice()
    print("USB speed:", device.getUsbSpeed().name)
    left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)
    left.requestOutput((640, 400)).link(stereo.left)
    right.requestOutput((640, 400)).link(stereo.right)
    depthQ = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
    pipeline.start()
    frames = 0
    center = 0.0
    t0 = time.monotonic()
    while pipeline.isRunning() and time.monotonic() - t0 < RUN_SECONDS:
        if depthQ.has():
            depth = depthQ.get().getFrame()
            h, w = depth.shape
            center = depth[h // 2, w // 2] / 1000
            frames += 1
        else:
            time.sleep(0.005)
    print(f"RESULT: frames={frames} seconds={RUN_SECONDS} center={center:.2f} m")