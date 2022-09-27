import os
import sys
import motmetrics as mm
import numpy as np
from PIL import Image
import cv2

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from tracking.bytetrack import byte_tracker
from tracking.deepsort import deepsort
from tracking.sort import sort

PARENT = '/mnt/03efd287-6b0b-4111-b250-fb25cb1f326c/Marc_Felip_Pomes/hakaton/'

def create_tracker(tracker_type):
    """
    Create a tracker based on the tracker_type. The tracker_type can be either 'sort', 'deepsort' or 'bytetrack'. The
    tracker_type 'sort' is based on the SORT tracker. The tracker_type 'deepsort' is based on the DeepSORT tracker. The
    tracker_type 'bytetrack' is based on the Bytetrack tracker.
    :param tracker_type: the tracker type
    :return: the tracker
    """
    model_path = PARENT + '/tracking/deepsort/checkpoints/ckpt.t7'

    # create the tracker
    if tracker_type == 'sort':
        tracker = sort.Sort()

    elif tracker_type == 'bytetrack':
        tracker = byte_tracker.BYTETracker()

    elif tracker_type == 'deepsort':
        #tracker = deepsort.DeepSort(model_path=os.path.join('deepsort', 'checkpoints', 'ckpt.t7'))
        tracker = deepsort.DeepSort(model_path=model_path)

    else:
        raise AssertionError('tracker_type should be named: sort, bytetrack or deepsort')

    return tracker



class FruitTracker:
    def __init__(self, tracker_type, img_sz):
        self.tracker_type = tracker_type
        self.frame_num = 0
        self.all_detections = []
        self.actual_frame_dets = []
        self.img_sz = img_sz
        self.det_centers = []
        self.det_ids = []
        self.predictions = []
        self.all_tracking_predicitons = []
        self.tracker = create_tracker(tracker_type)
        self.frame = []
        self.new_frame = []
        self.all_apples = {}
        self.new_frame_dets = []

    def get_detections(self, detections, frame_num, frame):
        self.frame = frame
        self.frame_num = frame_num
        self.actual_frame_dets = detections
        self.all_detections.append(detections)
        self.new_frame_dets = []

    def track_detections_frame(self, detections, tracker_type, img_size):

        tracker = self.tracker
        results = {
            'ids': [],
            'bboxes': [],
            'appears': [],
            'scores': []
        }

        # if there are no detections in the frame, skip it (all zeros)
        if detections is None or len(detections) == 0:
            trackers = self.tracker.update(np.empty((0, 5)),img_info=img_size, img_size=img_size)

        # if there are detections in the frame, track them
        else:
            # update the tracker with the detections
            if tracker_type == 'sort':
                trackers = self.tracker.update(np.array(detections))

            elif tracker_type == 'bytetrack':
                trackers = self.tracker.update(np.array(detections), img_info=img_size, img_size=img_size)

            elif tracker_type == 'deepsort':
                trackers = self.tracker.update(np.array(detections), img_file_name=self.frame)


            for t in trackers:
                # prepare data to be all the same format for all trackers
                if tracker_type == 'bytetrack':
                    score = t.score
                    t_id = t.track_id
                    t = t.tlbr
                    t = np.append(t, t_id)

                self.det_centers.append((int((t[0] + t[2]) / 2), int((t[1] + t[3]) / 2)))
                self.det_ids.append(int(t[4]))
                results['bboxes'].append([int(t[0]), int(t[1]), int(t[2]), int(t[3])])
                results['ids'].append(int(t[4]))
                results['appears'].append(1)
                results['scores'].append(score)


        self.predictions.append(results)
        return self.det_centers, self.det_ids, self.predictions

    def get_all_apples(self, all_apples):
        self.all_apples = all_apples

    def visualize_tracking_results(self):
        '''
        guardar frames videos
        '''
        video_results = PARENT + "/generatedFrames/prova1/"

        #cv2.putText(self.frame, ("Frame: " + str(self.frame_num)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
        #            3)
        cv2.putText(self.frame, ("Total Apples: " + str(len(self.all_apples))), (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 255), 3)

        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 255, 255)
            , (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (0, 128, 128), (128, 0, 128)
            , (128, 128, 128)]

        #for bbox, id in zip(self.all_tracking_predictions[self.frame_num]['bboxes'], self.all_tracking_predictions[self.frame_num]['ids']):
        for bbox, id, score in zip(self.all_tracking_predictions[self.frame_num]['bboxes'],
                                self.all_tracking_predictions[self.frame_num]['ids'],
                                self.all_tracking_predictions[self.frame_num]['scores']):
            color = colors[id % len(colors)]

            # get the bounding box coordinates
            x_min, y_min, x_max, y_max = bbox

            #get score
            #score = self.all_tracking_predicitons[self.frame_num][id]

            # plot the bounding box: prediction in red
            cv2.rectangle(self.frame, (x_min, y_min), (x_max, y_max), color, 2)

            cv2.putText(self.frame, str(round(score, 2)), (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        #cv2.imwrite(video_results + str(self.frame_num) + '.png', self.frame)
        #cv2.imshow('img', self.frame)
        #key = cv2.waitKey(0)
        return self.frame

    def track_yolo_results(self, partition='test', tracker_evaluation=True,
                           visualize_results=False, save_results=False):

        # where will be stored the predictions of the tracker
        # results of the tracking (metrics)
        tracker = self.tracker
        all_detections = self.all_detections

        #perform the tracking of the detections
        self.det_centers, self.det_ids, self.all_tracking_predictions = self.track_detections_frame(detections=self.actual_frame_dets,
                                                                                tracker_type=self.tracker_type,
                                                                                img_size=self.img_sz)
        #Crida a la funció per a visualtzar els resultats del tracking
        self.visualize_tracking_results()

        return self.all_tracking_predictions, self.frame

    def get_appearances(self, tracking_predictions):
        self.all_tracking_predicitons = tracking_predictions