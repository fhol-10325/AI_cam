import cv2
import argparse
from hand_landmarks_tflite import HandTracker
import time

# USAGE: python run.py --3d [true/false]
ap = argparse.ArgumentParser()
ap.add_argument("--3d", required=False,
	help="Check for type of detection")
args = vars(ap.parse_args())

WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "Models/palm_detection_without_custom_layer.tflite"
LANDMARK_MODEL_PATH = "Models/hand_landmark_3d.tflite"
ANCHORS_PATH = "anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture('Mediacpy/yoga-1.mp4')

if capture.isOpened():
    hasFrame, frame = capture.read()
else:
    hasFrame = False

#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (0, 9), (0, 13), (0, 17)
]

hand_3d = "True"

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1
)

while hasFrame:
    start_time = time.time()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = detector(image)
    if points is not None:
        if hand_3d == "True":
            for point in points:
                x, y = point
                cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            for connection in connections:
                x0, y0 = points[connection[0]]
                x1, y1 = points[connection[1]]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
        else:
            cv2.line(frame, (int(bbox[0][0]), int(bbox[0][1])), (int(bbox[1][0]), int(bbox[1][1])), CONNECTION_COLOR, THICKNESS)
            cv2.line(frame, (int(bbox[1][0]), int(bbox[1][1])), (int(bbox[2][0]), int(bbox[2][1])), CONNECTION_COLOR, THICKNESS)
            cv2.line(frame, (int(bbox[2][0]), int(bbox[2][1])), (int(bbox[3][0]), int(bbox[3][1])), CONNECTION_COLOR, THICKNESS)
            cv2.line(frame, (int(bbox[3][0]), int(bbox[3][1])), (int(bbox[0][0]), int(bbox[0][1])), CONNECTION_COLOR, THICKNESS)

    elapsed_time = 1000*(time.time() - start_time)
    elapsed_time = 1000/elapsed_time

    cv2.putText(frame, "Elapsed Time : " + '{:.1f}'.format(elapsed_time) + "fps",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4,
               cv2.LINE_AA)
    
    cv2.imshow(WINDOW, frame)
    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()