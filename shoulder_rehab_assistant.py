# -*- coding: utf-8 -*-
"""
Posture Assessment Tool – Frontal and Sagittal Planes
Created on Sun Aug 10 17:10:22 2025
@author: joono
"""

import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Helper functions
def calculate_angle(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.degrees(math.atan2(dy, dx))

def calculate_distance(p1, p2, image_shape):
    h, w = image_shape[:2]
    dx = (p2.x - p1.x) * w
    dy = (p2.y - p1.y) * h
    return math.sqrt(dx**2 + dy**2)

# Start video capture
cap = cv2.VideoCapture(0)
mode = "frontal"  # Default mode

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame")
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        h, w, _ = image_bgr.shape

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

            feedback = ""
            color = (255, 255, 255)

            if mode == "frontal":
                # Shoulder angle
                shoulder_angle = calculate_angle(left_shoulder, right_shoulder)
                # Shoulder height difference
                height_diff = abs(left_shoulder.y - right_shoulder.y) * h

                # Feedback logic
                if abs(shoulder_angle) > 5 or height_diff > 20:
                    feedback = "Uneven shoulders detected!"
                    color = (0, 0, 255)
                else:
                    feedback = "Shoulders are level."
                    color = (0, 255, 0)

                # Draw shoulder line
                pt1 = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                pt2 = (int(right_shoulder.x * w), int(right_shoulder.y * h))
                cv2.line(image_bgr, pt1, pt2, color, 2)

                # Display metrics
                cv2.putText(image_bgr, f'Shoulder angle: {shoulder_angle:.2f}°', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image_bgr, f'Height diff: {height_diff:.1f} px', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image_bgr, feedback, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(image_bgr, "Mode: Frontal (Press 's' for Sagittal)", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            elif mode == "sagittal":
                # Internal rotation: elbow to wrist angle
                rotation_angle = calculate_angle(left_elbow, left_wrist)
                # Shoulder protraction: shoulder to hip horizontal distance
                protraction = abs(left_shoulder.x - left_hip.x) * w

                # Feedback logic
                if rotation_angle < -30 or protraction > 30:
                    feedback = "Internal rotation or protraction detected!"
                    color = (0, 0, 255)
                else:
                    feedback = "Shoulder alignment looks good."
                    color = (0, 255, 0)

                # Display metrics
                cv2.putText(image_bgr, f'Rotation angle: {rotation_angle:.2f}°', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image_bgr, f'Protraction: {protraction:.1f} px', (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(image_bgr, feedback, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(image_bgr, "Mode: Sagittal (Press 'f' for Frontal)", (10, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Posture Assessment', image_bgr)

        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            mode = "frontal"
        elif key == ord('s'):
            mode = "sagittal"

cap.release()
cv2.destroyAllWindows()
