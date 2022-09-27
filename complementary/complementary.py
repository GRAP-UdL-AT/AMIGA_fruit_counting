def get_all_detections(detector, frame):
    # get the detections and confidences for each frame
    detections, out_scores = detector.run(frame)

    # x1,y1,x2,y2
    for detection in detections:
        detection[0] = detection[0]
        detection[1] = detection[1]
        detection[2] = detection[0] + (detection[2])
        detection[3] = detection[1] + (detection[3])

    # augment the bbox size by 30%
    dets_conf = []
    for k in range(len(out_scores)):
        a = detections[k]
        augment_size_of_bboxes_in_crops(a)
        b = out_scores[k]
        a.append(b)

        dets_conf.append(a)
    return dets_conf

def augment_size_of_bboxes_in_crops(bbox_tlbr, percentage_to_augment=0.15, size_img=(1080, 1920)):
    """
    Augment the size of the bboxes.
    :param bbox_tlbr: the bbox in tlbr format
    :param percentage_to_augment: the percentage to augment, in percentage of the size of the image
    :param size_img: the size of the image
    :return: the bbox in tlbr format with the size augmented
    """
    height = bbox_tlbr[3] - bbox_tlbr[1]
    width = bbox_tlbr[2] - bbox_tlbr[0]

    # augment the size of the bbox
    bbox_tlbr[0] -= percentage_to_augment * width
    if bbox_tlbr[0] < 0:
        bbox_tlbr[0] = 0
    bbox_tlbr[1] -= percentage_to_augment * height
    if bbox_tlbr[1] < 0:
        bbox_tlbr[1] = 0

    bbox_tlbr[2] += percentage_to_augment * width
    if bbox_tlbr[2] > size_img[0]:
        bbox_tlbr[2] = size_img[0]

    bbox_tlbr[3] += percentage_to_augment * height
    if bbox_tlbr[3] > size_img[1]:
        bbox_tlbr[3] = size_img[1]

    return [int(x) for x in bbox_tlbr]


def perform_tracking(new_predictions, all_apples):
    for id in new_predictions['ids']:
        # get the index of each id
        apple_id_index = new_predictions['ids'].index(id)

        # if the ID is not in the dict, save it
        if id not in all_apples:
            new_id = new_predictions['ids'][apple_id_index]
            all_apples.append(new_id)

    return all_apples

def resize_img(frame):
    frame = frame[300:900:,:]
    return frame