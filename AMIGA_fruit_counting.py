# Copyright (c) farm-ng, inc. All rights reserved.
import argparse
import asyncio

import cv2
import numpy as np
from farm_ng.oak import oak_pb2
from farm_ng.oak.client import OakCameraClient
from farm_ng.oak.client import OakCameraClientConfig

import cv2
import os
from yolov5.detect_pol_v2 import yolo_detector
from complementary import complementary
from FruitTracker import FruitTracker

#python3 main_fruit_counting.py --port 50051 --address 192.168.0.189


CURRENT = os.path.dirname(os.path.realpath(__file__))
PARENT = os.path.dirname(CURRENT)

async def main(address: str, port: int, stream_every_n: int) -> None:
    # configure the camera client
    config = OakCameraClientConfig(address=address, port=port)
    client = OakCameraClient(config)

    # get the streaming object
    response_stream = client.stream_frames(every_n=stream_every_n)

    # start the streaming service
    await client.start_service()

    # get all variables to pass to tHe tracker
    imgsz = (1080, 1920)
    #imgsz *= 2 if len(imgsz) == 1 else 1
    data = PARENT + '/camera_client/yolov5/data/segmentacio_pomes.yaml'
    conf_thres = 0.15
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

    cv2.namedWindow("Farm@thon", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Farm@thon", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    i = 0

    key = ''
    #while True:
    while key != 113:
        # query the service state
        state: oak_pb2.OakServiceState = await client.get_state()

        if state.value != oak_pb2.OakServiceState.RUNNING:
            print("Camera is not streaming!")
            continue

        response: oak_pb2.StreamFramesReply = await response_stream.read()
        if response and response.status == oak_pb2.ReplyStatus.OK:
            # get the sync frame
            frame: oak_pb2.OakSyncFrame = response.frame
            print(f"Got frame: {frame.sequence_num}")
            print(f"Device info: {frame.device_info}")
            print("#################################\n")

            # get the image data and decode
            data: bytes = getattr(frame, "rgb").image_data

            # use imdecode function
            image = np.frombuffer(data, dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)

            '''
            Codi TRACKING
            '''
            frame = image
            frame = cv2.resize(frame, (960,640))

            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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

            #cv2.imshow("Farm@thon", cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE))
            cv2.imshow("Farm@thon", frame)

            key = cv2.waitKey(5)

            '''
            # visualize the image
            cv2.namedWindow("rgb", cv2.WINDOW_NORMAL)
            cv2.imshow("rgb", image)
            cv2.waitKey(1)
            '''

        i += 1
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="amiga-camera-app")
    parser.add_argument("--port", type=int, required=True, help="The camera port.")
    parser.add_argument("--address", type=str, default="localhost", help="The camera address")
    parser.add_argument("--stream-every-n", type=int, default=4, help="Streaming frequency")
    args = parser.parse_args()

    asyncio.run(main(args.address, args.port, args.stream_every_n))
