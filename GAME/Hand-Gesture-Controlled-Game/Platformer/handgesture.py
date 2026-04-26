# handgesture.py

import cv2
import mediapipe as mp

class HandGestureController:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to initialize camera.")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

    def get_gesture(self):
        success, img = self.cap.read()
        if not success:
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist_x = hand_landmarks.landmark[0].x
                index_tip_y = hand_landmarks.landmark[8].y
                index_pip_y = hand_landmarks.landmark[6].y
                pinky_root_x = hand_landmarks.landmark[17].x
                thumb_root_x = hand_landmarks.landmark[5].x

                # Detect gestures
                if wrist_x < pinky_root_x:
                    return "move_left"
                elif wrist_x > thumb_root_x:
                    return "move_right"
                elif index_tip_y < index_pip_y:
                    return "jump"

        return None

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
