import depthai as dai

with dai.Pipeline() as pipeline:
        device = pipeline.getDefaultDevice()
        print("Name: ", device.getDeviceName())
        print("USB speed:", device.getUsbSpeed().name)
        print("Cameras: ", device.getConnectedCameras())