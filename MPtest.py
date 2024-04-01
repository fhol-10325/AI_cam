import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Solutions
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# For webcam input, create a VideoCapture object
cap = cv2.VideoCapture(0)

prev_frame_time = 0
curr_frame_time = 0


with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:  # Record for 10 seconds
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            break

        curr_frame_time = time.time()

        # Convert the BGR image to RGB, flip the image around y-axis for correct handedness output
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, mark the image as not writeable to pass by reference
        image.flags.writeable = False
        hand_results = hands.process(image)
        pose_results = pose.process(image)

        # Draw the hand and pose annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Draw pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        

        fps = 1 / (curr_frame_time - prev_frame_time)
        prev_frame_time = curr_frame_time

        # Display the FPS on the frame
        cv2.putText(image, f'FPS: {int(fps)}', (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('MediaPipe Hands and Pose', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
            break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()