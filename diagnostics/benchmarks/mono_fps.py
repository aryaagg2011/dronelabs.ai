import sys
import time
import depthai as dai

SOCKETS = {
    "A": dai.CameraBoardSocket.CAM_A,
    "B": dai.CameraBoardSocket.CAM_B,
    "C": dai.CameraBoardSocket.CAM_C,
}

choice = sys.argv[1].upper() if len(sys.argv) > 1 else "B"
fps = float(sys.argv[2]) if len(sys.argv) > 2 else None
seconds = 15

with dai.Pipeline() as pipeline:
    queues = {}
    for ch in choice:
        cam = pipeline.create(dai.node.Camera).build(SOCKETS[ch])
        if fps:
            out = cam.requestOutput((640, 400), fps=fps)
        else:
            out = cam.requestOutput((640, 400))
        queues[ch] = out.createOutputQueue()
    pipeline.start()

    counts = {ch: 0 for ch in queues}
    start = time.time()
    last = start
    while pipeline.isRunning() and time.time() - start < seconds:
        for ch, q in queues.items():
            q.get()
            counts[ch] += 1
        if time.time() - last >= 1:
            last = time.time()
            print(f"{time.time() - start:4.1f}s frames: {counts}")
    print(f"Ended after {time.time() - start:.1f}s")