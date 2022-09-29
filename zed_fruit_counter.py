import pyzed.sl as sl
import cv2
import os

from yolov5.detect_pol_v2 import yolo_detector
from complementary import complementary
from FruitTracker import FruitTracker

CURRENT = os.path.dirname(os.path.realpath(__file__))
PARENT = os.path.dirname(CURRENT)

camera_settings = sl.VIDEO_SETTINGS.BRIGHTNESS
str_camera_settings = "BRIGHTNESS"
step_camera_settings = 1

VIDEO_PATH1 = PARENT + '/generatedVideo/frame1.png'
VIDEO_PATH2 = PARENT + '/generatedVideo/frame2.png'

def main():
    print("Running...")
    init = sl.InitParameters()
    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening ZED Camera...")
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    #Setting initParams
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.camera_fps = 15

    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    print_help()

    # get all variables to pass to tHe tracker
    imgsz = [640]
    imgsz *= 2 if len(imgsz) == 1 else 1
    data = PARENT + '/code/yolov5/data/segmentacio_pomes.yaml'
    conf_thres = 0.5
    iou_thres = 0.45
    half = False
    dnn = False
    max_det = 1000
    # weights_path = PARENT + '/data/weights/best.pt'
    weights_path = PARENT + '/data/weights/yolov5x_finetuned_12356.pt'
    # weights_path = PARENT + '/data/weights/yolov5s_finetuned_12356.pt'
    # weights_path = PARENT + '/data/weights/yolov5x_12356.pt'

    # Create detector and pass variables
    detector = yolo_detector(weights_path, imgsz, data, conf_thres, iou_thres, half, dnn, max_det)

    # Create Tracker and pass the tracker type
    fruit_tracker = FruitTracker(tracker_type='bytetrack', img_sz=imgsz)

    # List to store apples IDs
    all_apples = []

    cv2.namedWindow("ZED", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("ZED", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    i = 0
    key = ''
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            frame = mat.get_data()

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # before passing the frame to the detector, we must delete the last chanel, due the detector works with 3 channels images
            if len(frame.shape) > 2 and frame.shape[2] == 4:
                # convert the image from RGBA2RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            #Get just a part of the frame not the full frame
            frame = complementary.resize_img(frame)

            # Call the detector and get all de the detections
            dets_conf = complementary.get_all_detections(detector, frame)

            # pass the detections, number of frame and frame(np.array) to the tracker, also pass camera name
            fruit_tracker.get_detections(dets_conf, i, frame)

            # pass the number of apples
            fruit_tracker.get_all_apples(all_apples)

            # perform the tracking and save the results, new_predictions = last predictions returned by tracker
            tracking_predictions, img = fruit_tracker.track_yolo_results()
            new_predictions = tracking_predictions[-1]

            all_apples = complementary.perform_tracking(new_predictions, all_apples)

            cv2.imshow("ZED", frame)

            key = cv2.waitKey(5)
            settings(key, cam, runtime, mat)
        else:
            key = cv2.waitKey(5)
        i += 1
    cv2.destroyAllWindows()

    cam.close()
    print("\nFINISH")

def print_help():
    print("Help for camera setting controls")
    print("  Increase camera settings value:     +")
    print("  Decrease camera settings value:     -")
    print("  Switch camera settings:             s")
    print("  Reset all parameters:               r")
    print("  Record a video:                     z")
    print("  Quit:                               q\n")

def settings(key, cam, runtime, mat):
    if key == 43:  # for '+' key
        current_value = cam.get_camera_settings(camera_settings)
        cam.set_camera_settings(camera_settings, current_value + step_camera_settings)
        print(str_camera_settings + ": " + str(current_value + step_camera_settings))
    elif key == 45:  # for '-' key
        current_value = cam.get_camera_settings(camera_settings)
        if current_value >= 1:
            cam.set_camera_settings(camera_settings, current_value - step_camera_settings)
            print(str_camera_settings + ": " + str(current_value - step_camera_settings))
    elif key == 114:  # for 'r' key
        cam.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.CONTRAST, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.HUE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, -1)
        cam.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_TEMPERATURE, -1)
        print("Camera settings: reset")

if __name__ == "__main__":
    main()