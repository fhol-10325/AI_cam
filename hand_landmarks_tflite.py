import cv2
import tensorflow as tf
import copy
import numpy as np
import csv

def non_max_suppression_fast(boxes, probabilities=None, overlap_threshold=0.3):
    """
    Algorithm to filter bounding box proposals by removing the ones with a too low confidence score
    and with too much overlap.

    Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

    :param boxes: List of proposed bounding boxes
    :param overlap_threshold: the maximum overlap that is allowed
    :return: filtered boxes
    """
    # if there are no boxes, return an empty list
    if boxes.shape[1] == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] - (boxes[:, 2] / [2])  # center x - width/2
    y1 = boxes[:, 1] - (boxes[:, 3] / [2])  # center y - height/2
    x2 = boxes[:, 0] + (boxes[:, 2] / [2])  # center x + width/2
    y2 = boxes[:, 1] + (boxes[:, 3] / [2])  # center y + height/2

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = boxes[:, 2] * boxes[:, 3]  # width * height
    idxs = y2


    # if probabilities are provided, sort on them instead
    if probabilities is not None:
        idxs = probabilities

    # sort the indexes
    idxs = np.argsort(idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))
    # return only the bounding boxes that were picked
    return pick

def inference(img, input_size, interpreter):

     # reading the SSD anchors
    anchors_path = 'anchors.csv'
    with open(anchors_path, "r") as csv_f:
        anchors = np.r_[
            [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
        ]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, dsize=(input_size, input_size))
    img = img.astype(np.float32)
    img = (img / 128) - 1  # Normalize to [-0.5, 0.5]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_idx = input_details[0]['index']
    out_reg_idx = output_details[0]['index']
    out_clf_idx = output_details[1]['index']
    #print(output_details)

    interpreter.set_tensor(in_idx, img[None])
    interpreter.invoke()

    out_reg = interpreter.get_tensor(out_reg_idx)[0]
    out_clf = interpreter.get_tensor(out_clf_idx)[0,:,0]

    P = 1/(1 + np.exp(-out_clf))
    detecion_mask = P > 0.5
    candidate_detect = out_reg[detecion_mask]
    print(np.shape(anchors[2016]))
    candidate_anchors = anchors[detecion_mask]
    P = P[detecion_mask]

    moved_candidate_detect = candidate_detect.copy()
    moved_candidate_detect[:, :2] = candidate_detect[:, :2] + (candidate_anchors[:, :2] * 256)
    box_ids = non_max_suppression_fast(moved_candidate_detect[:, :4], P)

    # Pick the first detected hand. Could be adapted for multi hand recognition
    box_ids = box_ids[0]

    # bounding box offsets, width and height
    dx,dy,w,h = candidate_detect[box_ids, :4]
    center_wo_offst = candidate_anchors[box_ids,:2] * 256

    # 7 initial keypoints
    keypoints = center_wo_offst + candidate_detect[box_ids,4:].reshape(-1,2)
    side = max(w,h) * 1.5

    print(keypoints[0], keypoints[1])

    return dx, dy


cap_device = 0
keypoint_score_th = 0.2

cap_device = 'Mediacpy/dance.mp4'
cap = cv2.VideoCapture(cap_device)

cv2.namedWindow('image')

model_path = 'Models/palm_detection_lite.tflite'
input_size = 192

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

ret, frame = cap.read()

img = copy.deepcopy(frame)

dx, dy = inference(img, input_size, interpreter)

cv2.circle(img, (int(196+dx*196), int(dy*196)), 4, (255, 255, 0), -1)



while True:
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

    cv2.imshow('image', img)
