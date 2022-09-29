import pyzed.sl as sl
from pyk4a import PyK4APlayback
import cv2
import os
import time

from yolov5.detect_pol_v2 import yolo_detector
from complementary import complementary
from FruitTracker import FruitTracker

CURRENT = os.path.dirname(os.path.realpath(__file__))
PARENT = os.path.dirname(CURRENT)

VIDEO_PATH = PARENT + '/generatedVideo/video6.mp4'
VIDEO_DURATION = 20
FRAME_RATE = 30

def main():

    #Setting initParams
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = FRAME_RATE

    zed = sl.Camera()
    if not zed.is_opened():
        print("Opening ZED Camera...")
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    i = 0   #Mod

    #Save video
    size = (1920, 1080)
    rotated_size = (1080, 1920)

    #video = cv2.VideoWriter(VIDEO_PATH,cv2.VideoWriter_fourcc(*'MP4V'),15, rotated_size)

    video = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), FRAME_RATE, rotated_size)

    start_time = time.time()
    #key = ''
    while i < 360:

        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:

            #get the image
            zed.retrieve_image(mat, sl.VIEW.LEFT)
            #key = cv2.waitKey(5)
            #settings(key, zed, runtime, mat)
            frame = mat.get_data()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            if len(frame.shape) > 2 and frame.shape[2] == 4:
                # convert the image from RGBA2RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            video.write(frame)

        else:
            key = cv2.waitKey(5)

        i += 1

    video.release()
    cv2.destroyAllWindows()

    zed.close()
    print("\nFINISH")

if __name__ == "__main__":
    main()
