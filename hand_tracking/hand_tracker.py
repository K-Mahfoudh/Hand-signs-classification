import cv2
import time
import mediapipe as mp
import numpy as np
from data import preprocess_image

BLUE_COLOR = (87, 53, 29)
RED_COLOR = (70, 57, 230)


class HandTracker:
    """
    Class used to detect and track hands using MediaPipe framework.

    """
    def __init__(self, network):
        self.network = network
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                         max_num_hands=2,
                                         min_detection_confidence=0.5,
                                         min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.pTime = 0

    def capture_video(self, detect: bool):
        """
        A method used to capture video using computer's camera, then detect and track hands, and finally classify
        detected hand's image using CNN based model.

        """
        frame_interval = 30
        frame_count = 0
        while True:
            success, frame = self.cap.read()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detecting hands
            detection_output = self.detect_hands(image_rgb, frame)

            # Getting hands bounding boxes
            if detection_output.multi_hand_landmarks and detect:
                bbox = get_bounding_box(frame, detection_output)
                # Drawing bounding box
                min_x, min_y, max_x, max_y = bbox
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), BLUE_COLOR, 3)

                # cropping the image to extract the hands based on their bounding boxes
                if (frame_count % frame_interval) == 0:
                    cropped_image = image_rgb[min_y:max_y, min_x:max_x]
                    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
                    image_tensor = preprocess_image(cropped_image_rgb)
                    if image_tensor:
                        result = self.network.predict_single(image_tensor)
                        print(result)
                    frame_count = 0

            frame_count += 1
            c_time = time.time()
            fps = 1 / (c_time - self.pTime)
            self.pTime = c_time
            cv2.putText(frame, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE_COLOR, 2)
            cv2.imshow("Test", frame)
            cv2.waitKey(1)

    def detect_hands(self, image_rgb, frame):
        """
        Method used for hands detection.

        :param image_rgb: original frame converted to RGB image (using cv2.cvtColor())
        :param frame: the original frame detected with opencv
        :return: Detection result containing hand landmarks
        """
        detection_output = self.hands.process(image_rgb)
        if detection_output.multi_hand_landmarks:
            for hand_lms in detection_output.multi_hand_landmarks:
                for _, lm in enumerate(hand_lms.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 3, RED_COLOR, cv2.FILLED)

                self.mp_draw.draw_landmarks(frame, hand_lms, self.mp_hands.HAND_CONNECTIONS)
        return detection_output


def get_bounding_box(image, results, padding=45):
    """
    Function used to get hand's bounding box based on its landmarks.

    :param image: the original frame detected using opencv
    :param results: the result of hands landmark detection
    :param padding: padding to be added to the bounding box
    :return: a tuple of minimum and maximum hand's landmarks coordinates
    """

    # Create a copy of the input image to draw bounding boxes on and write hands types labels.
    height, width, _ = image.shape

    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

        # Initialize a list to store the detected landmarks of the hand.
        landmarks = []

        # Iterate over the detected landmarks of the hand.
        for landmark in hand_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

        # Getting the X coords
        x_coordinates = np.array(landmarks)[:, 0]

        # Getting the Y coords
        y_coordinates = np.array(landmarks)[:, 1]

        # Get the bounding box coordinates for the hand with the specified padding.
        min_x = int(np.min(x_coordinates) - padding) if int(np.min(x_coordinates) - padding) > 0 else 0
        min_y = int(np.min(y_coordinates) - padding) if int(np.min(y_coordinates) - padding) > 0 else 0
        max_x = int(np.max(x_coordinates) + padding) if int(np.max(x_coordinates) + padding) > 0 else 0
        max_y = int(np.max(y_coordinates) + padding) if int(np.max(y_coordinates) + padding) > 0 else 0

        return min_x, min_y, max_x, max_y
