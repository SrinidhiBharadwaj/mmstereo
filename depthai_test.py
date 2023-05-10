import os
import cv2
import numpy as np
import argparse
import depthai as dai

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fps', type=int, default=4)
    return parser.parse_args()

def main():
    flags = parse_args()

    NN_INPUT_SHAPE = (400, 640)
    TARGET_SHAPE = 400, 640

    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2022_1)

    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)

    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_left.setFps(flags.fps)

    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_right.setFps(flags.fps)

    nn = pipeline.createNeuralNetwork()
    nn.input.setQueueSize(1)
    nn.setNumInferenceThreads(2)
    nn.setNumPoolFrames(2)
    nn.input.setBlocking(False)
    nn.setBlobPath('model.blob')

    mono_left.out.link(nn.inputs['left_input'])
    mono_right.out.link(nn.inputs['right_input'])

    nn_xout = pipeline.createXLinkOut()
    nn_xout.setStreamName('disparity')
    nn.out.link(nn_xout.input)

    with dai.Device(pipeline) as device:
        device.setLogLevel(dai.LogLevel.INFO)
        queue_nn = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)

        model_max_disp = 256
        nn_disp_multiplier =  255.0 / model_max_disp
        scale_multiplier = TARGET_SHAPE[1] / NN_INPUT_SHAPE[1]

        while True:
            fps_handler.tick('disparity')

            nn_msg = queue_nn.get()

            disparity = np.array(nn_msg.getLayerFp16('disparity'))
            nn_output = disparity.reshape(TARGET_SHAPE)
            nn_disp = (nn_output * nn_disp_multiplier * scale_multiplier).astype(np.uint8)

            disparity_vis = cv2.applyColorMap(nn_disp, cv2.COLORMAP_INFERNO)

            cv2.imshow("Disparity", disparity_vis)

            if cv2.waitKey(1) == ord('q'):
                break

if __name__ == "__main__":
    main()
