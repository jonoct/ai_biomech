# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 19:10:54 2025

@author: joono
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
import sys
from collections import deque

# =========================
# Config
# =========================
HISTORY_LEN = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX
VEL_THRESH = 0.02
RECORD_DURATION = 10  # seconds
COUNTDOWN = 3         # seconds before recording starts

# Punches and views
PUNCHES = ["Jab", "Cross", "Uppercut", "Hook"]
VIEWS = ["Frontal", "Sagittal"]

# =========================
# MediaPipe setup
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# =========================
# Geometry helpers
# =========================
def clamp(x, a, b):
    return max(a, min(b, x))

def angle_at_point(a, b, c):
    ba = a - b
    bc = c - b
    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)
    if na == 0 or nc == 0:
        return np.nan
    cosang = np.dot(ba, bc) / (na * nc)
    cosang = clamp(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def np_point(lm):
    return np.array([lm.x, lm.y], dtype=np.float32)

def to_px(pt, w, h):
    return int(pt[0] * w), int(pt[1] * h)

def put_label(img, text, org, color, scale=0.6, thickness=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)

# =========================
# Angle extraction
# =========================
def extract_joint_angles(lm):
    """Return dict of key joint angles for arms, trunk, legs, hands."""
    # Points
    L = mp_pose.PoseLandmark
    pts = {name: np_point(lm[getattr(L, name).value]) for name in L.__members__}

    angles = {}
    # Arms
    angles['L_elbow'] = angle_at_point(pts['LEFT_SHOULDER'], pts['LEFT_ELBOW'], pts['LEFT_WRIST'])
    angles['R_elbow'] = angle_at_point(pts['RIGHT_SHOULDER'], pts['RIGHT_ELBOW'], pts['RIGHT_WRIST'])
    angles['L_shoulder'] = angle_at_point(pts['LEFT_HIP'], pts['LEFT_SHOULDER'], pts['LEFT_ELBOW'])
    angles['R_shoulder'] = angle_at_point(pts['RIGHT_HIP'], pts['RIGHT_SHOULDER'], pts['RIGHT_ELBOW'])

    # Trunk
    angles['trunk_forward'] = angle_at_point(pts['LEFT_HIP'], pts['LEFT_SHOULDER'], pts['RIGHT_HIP'])
    angles['trunk_sidebend'] = angle_at_point(pts['RIGHT_HIP'], pts['RIGHT_SHOULDER'], pts['LEFT_HIP'])

    # Legs
    angles['L_knee'] = angle_at_point(pts['LEFT_HIP'], pts['LEFT_KNEE'], pts['LEFT_ANKLE'])
    angles['R_knee'] = angle_at_point(pts['RIGHT_HIP'], pts['RIGHT_KNEE'], pts['RIGHT_ANKLE'])
    angles['L_hip'] = angle_at_point(pts['LEFT_SHOULDER'], pts['LEFT_HIP'], pts['LEFT_KNEE'])
    angles['R_hip'] = angle_at_point(pts['RIGHT_SHOULDER'], pts['RIGHT_HIP'], pts['RIGHT_KNEE'])

    # Ankles
    angles['L_ankle'] = angle_at_point(pts['LEFT_KNEE'], pts['LEFT_ANKLE'], pts['LEFT_FOOT_INDEX'])
    angles['R_ankle'] = angle_at_point(pts['RIGHT_KNEE'], pts['RIGHT_ANKLE'], pts['RIGHT_FOOT_INDEX'])

    # Hands (wrist flexion relative to forearm)
    angles['L_wrist'] = angle_at_point(pts['LEFT_ELBOW'], pts['LEFT_WRIST'], pts['LEFT_INDEX'])
    angles['R_wrist'] = angle_at_point(pts['RIGHT_ELBOW'], pts['RIGHT_WRIST'], pts['RIGHT_INDEX'])

    return angles

# =========================
# Recording workflow
# =========================
def record_sequence(cap, pose, punch, view, lead_hand):
    """Handles countdown, recording, and returns DataFrame of angles."""
    if not cap.isOpened():
        sys.exit("❌ Could not open webcam. Check if it's in use by another app or if permissions are blocked.")

    # Create main display window
    cv2.namedWindow("Punch Recording", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Punch Recording", cv2.WND_PROP_TOPMOST, 1)

    data = []
    recording = False
    countdown_start = None
    record_start = None

    # --- Pre-record control screen ---
    while True:
        ctrl_img = np.zeros((200, 500, 3), dtype=np.uint8)
        cv2.putText(ctrl_img, f"{punch} ({view})", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(ctrl_img, "Press SPACE to start countdown", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(ctrl_img, "Press Q to quit", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Punch Recording", ctrl_img)
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACE
            countdown_start = time.time()
            break
        elif key in (ord('q'), ord('Q'), 27):
            return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame capture failed — ending recording.")
            break
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            angles = extract_joint_angles(results.pose_landmarks.landmark)

            # Overlay angles
            y_off = 20
            for k, v in angles.items():
                put_label(frame, f"{k}: {v:.1f}", (10, y_off), (255, 255, 255))
                y_off += 18

            if recording:
                elapsed = time.time() - record_start
                if elapsed <= RECORD_DURATION:
                    row = {'time': elapsed, **angles}
                    data.append(row)
                    put_label(frame, f"Recording {punch} ({view}) {elapsed:.1f}s",
                              (10, h - 20), (0, 0, 255), scale=1)
                else:
                    break

        # Countdown display
        if countdown_start and not recording:
            cd_elapsed = time.time() - countdown_start
            remaining = COUNTDOWN - int(cd_elapsed)
            if remaining > 0:
                put_label(frame, f"Starting in {remaining}",
                          (w//2 - 50, h//2), (0, 255, 255), scale=2)
            else:
                recording = True
                countdown_start = None
                record_start = time.time()

        cv2.imshow("Punch Recording", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            return None

    return pd.DataFrame(data)

# =========================
# Main program
# =========================
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("❌ Could not open webcam.")

    lead_hand = None
    while lead_hand not in ['left', 'right']:
        lead_hand = input("Select lead hand (left/right): ").strip().lower()

    results_dict = {}
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for punch in PUNCHES:
            for view in VIEWS:
                while True:
                    df = record_sequence(cap, pose, punch, view, lead_hand)
                    if df is None:
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                    # --- Post-record control screen ---
                    ctrl_img = np.zeros((200, 500, 3), dtype=np.uint8)
                    cv2.putText(ctrl_img, f"{punch} ({view}) done!", (20, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(ctrl_img, "Press R to retry", (20, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(ctrl_img, "Press N for next", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(ctrl_img, "Press Q to quit", (20, 170),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("Punch Recording", ctrl_img)
                    choice = None
                    while choice is None:
                        key = cv2.waitKey(0) & 0xFF
                        if key in (ord('r'), ord('R')):
                            choice = 'r'
                        elif key in (ord('n'), ord('N')):
                            choice = 'n'
                        elif key in (ord('q'), ord('Q'), 27):
                            choice = 'q'

                    if choice == 'r':
                        continue
                    elif choice == 'n':
                        results_dict[f"{punch}_{view}"] = df
                        break
                    elif choice == 'q':
                        cap.release()
                        cv2.destroyAllWindows()
                        return

        # Export CSVs
        for key, df in results_dict.items():
            filename = f"{key.replace(' ', '_')}.csv"
            df.to_csv(filename, index=False)
            print(f"💾 Saved {filename}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
