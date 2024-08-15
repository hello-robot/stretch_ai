# Copyright (c) Hello Robot, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in the root directory
# of this source tree.
#
# Some code may be adapted from other open-source works with their respective licenses. Original
# license information maybe found below, if so.

from enum import Enum
from typing import List

import cv2
import mediapipe as mp
import numpy as np
from google.protobuf.json_format import MessageToDict

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

from webcam import Webcam

LANDMARK = mp.solutions.hands.HandLandmark


class Direction(Enum):
    NONE = 0
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


############################################################
# helper functions
def _tip2wrist_dist(tip: LANDMARK, estimate) -> float:
    tip_landmark = estimate[0].landmark[tip.value]
    wrist_landmark = estimate[0].landmark[LANDMARK.WRIST.value]

    t = np.array([tip_landmark.x, tip_landmark.y, tip_landmark.z])
    w = np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z])

    dist = np.linalg.norm(t - w)
    return dist


def _tip2wrist_direction(tip: LANDMARK, estimate):
    tip_landmark = estimate[0].landmark[tip.value]
    wrist_landmark = estimate[0].landmark[LANDMARK.WRIST.value]

    t = np.array([tip_landmark.x, tip_landmark.y, tip_landmark.z])
    w = np.array([wrist_landmark.x, wrist_landmark.y, wrist_landmark.z])
    v = t - w

    angle = np.arctan2(v[0], v[1])
    return angle


def _tip2tip_distance(estimate, finger_1, finger_2):
    f1_landmark = estimate[0].landmark[finger_1.value]
    f2_landmark = estimate[0].landmark[finger_2.value]

    f1 = np.array([f1_landmark.x, f1_landmark.y, f1_landmark.z])
    f2 = np.array([f2_landmark.x, f2_landmark.y, f2_landmark.z])

    return np.linalg.norm(f2 - f1)


def _get_extended_fingers(estimate, threshold=0.25):
    tips = [
        LANDMARK.THUMB_TIP,
        LANDMARK.INDEX_FINGER_TIP,
        LANDMARK.MIDDLE_FINGER_TIP,
        LANDMARK.RING_FINGER_TIP,
        LANDMARK.PINKY_TIP,
    ]

    distances = [_tip2wrist_dist(tip, estimate) for tip in tips]

    extended = []
    for tip, distance in zip(tips, distances):
        if distance > threshold:
            extended.append(tip)

    return extended


def _compute_bounding_box(hand_prediction_results, min_size=0.1):
    landmarks = hand_prediction_results.multi_hand_landmarks
    if landmarks is not None:
        x_min = y_min = 1
        x_max = y_max = 0
        for landmark in landmarks[0].landmark:
            x, y = landmark.x, landmark.y
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # make bounding box square
        dx = x_max - x_min
        dy = y_max - y_min

        # enforce min size
        dx = max(dx, min_size)
        dy = max(dy, min_size)

        # make bounding box square
        if dx > dy:
            y_max = y_min + dx
        else:
            x_max = x_min + dy
        return x_min, y_min, x_max, y_max


def _normalize_estimate(hand_prediction_results):
    landmarks = hand_prediction_results.multi_hand_landmarks
    x_min, y_min, x_max, y_max = _compute_bounding_box(hand_prediction_results)

    if landmarks is not None:
        for landmark in landmarks[0].landmark:
            x, y = landmark.x, landmark.y
            x_norm = (x - x_min) / (x_max - x_min)
            y_norm = (y - y_min) / (y_max - y_min)
            landmark.x = x_norm
            landmark.y = y_norm

    return hand_prediction_results


############################################################
class HandAnalyzer:
    # contains methods for analyzing the results of a hand pose estimate
    def __init__(self) -> None:
        pass

    def get_extended_fingers(self, hand_prediction_results, threshold=0.25) -> List[LANDMARK]:
        landmarks = hand_prediction_results.multi_hand_landmarks
        if landmarks is not None:
            return _get_extended_fingers(landmarks, threshold=threshold)
        return []

    def get_finger_direction(self, hand_prediction_results, finger: LANDMARK) -> float:
        landmarks = hand_prediction_results.multi_hand_landmarks
        if landmarks is not None:
            return _tip2wrist_direction(finger, landmarks)

        return 0.0

    def get_finger_distance(self, hand_prediction_results, finger_1: LANDMARK, finger_2: LANDMARK):
        landmarks = hand_prediction_results.multi_hand_landmarks
        if landmarks is not None:
            return _tip2tip_distance(landmarks, finger_1, finger_2)

        return 10.0

    def check_fingers_in_contact(
        self, hand_prediction_results, finger_1: LANDMARK, finger_2: LANDMARK, threshold=0.1
    ):
        if self.get_finger_distance(hand_prediction_results, finger_1, finger_2) < threshold:
            return True
        else:
            return False


class HandTracker(HandAnalyzer):
    # contains interface with a webcam and visualization methods
    def __init__(self, left_clutch: bool = True) -> None:
        self.left_clutch = left_clutch
        self.hands_model = mp_hands.Hands(
            model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

    def run_detection(self, image):
        image = cv2.flip(image, 0)
        hand_prediction_results = self.hands_model.process(image)
        return hand_prediction_results

    def check_clutched(self, hand_prediction_results, complicated: bool = False) -> bool:
        clutched = False
        if hand_prediction_results.multi_handedness is not None:
            for _, hand_handedness in enumerate(hand_prediction_results.multi_handedness):
                handedness_dict = MessageToDict(hand_handedness)
                hand_side = (handedness_dict["classification"][0]["label"]).lower()
                if self.left_clutch and hand_side == "left":
                    clutched = True
                elif not self.left_clutch and hand_side == "right":
                    clutched = True

        if complicated:
            # complicated method
            if hand_prediction_results.multi_hand_landmarks is not None:
                hand_prediction_results = _normalize_estimate(hand_prediction_results)

            # get extended fingers
            extended_fingers = self.get_extended_fingers(hand_prediction_results, threshold=0.5)

            # determine action
            n_fingers = len(extended_fingers)

            # five finger actions
            if n_fingers == 5:
                clutched = True

        return clutched

    def run(self):
        cam = Webcam(show_images=False, use_second_camera=False)

        while True:
            image, _ = cam.get_next_frame()
            detection_result = self.run_detection(image)
            clutched = self.check_clutched(detection_result)

            if clutched:
                print("CLUTCH ENGAGED")
            else:
                print("RUN")


if __name__ == "__main__":
    HandTracker(left_clutch=False).run()
