# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 23:17:06 2025

@author: joono
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 21:59:30 2025

@author: joono
"""

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
from datetime import datetime
import math

# Hardcoded CSV file name
CSV_FILE = "shoulder_posture_results.csv"

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# -----------------------------
# Utility Functions
# -----------------------------
def get_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle

def log_result(assessment, metrics, classification):
    with open(CSV_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, assessment, metrics, classification])

def show_countdown(frame, window_name="Shoulder Posture Assessment"):
    """Overlay 3-2-1 countdown before starting assessment"""
    for i in range(3, 0, -1):
        countdown_frame = frame.copy()
        cv2.putText(countdown_frame, str(i),
                    (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        cv2.imshow(window_name, countdown_frame)
        cv2.waitKey(1000)

# -----------------------------
# Assessments
# -----------------------------
def assess_cva(points):
    ear = points[mp_pose.PoseLandmark.LEFT_EAR.value]
    shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]
    cva = get_angle(ear, shoulder, hip)
    classification = "Forward Head" if cva < 48 else "Normal"
    return cva, classification

def assess_rounded_shoulders(points):
    shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = points[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    wrist = points[mp_pose.PoseLandmark.LEFT_WRIST.value]
    angle = get_angle(shoulder, elbow, wrist)
    classification = "Rounded Shoulders" if angle < 150 else "Normal"
    return angle, classification

def assess_scapular(points):
    l_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    diff = abs(l_shoulder[1] - r_shoulder[1])
    classification = "Asymmetry" if diff > 30 else "Normal"
    return diff, classification

def assess_shoulder_tilt(points):
    l_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    r_shoulder = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    tilt = l_shoulder[1] - r_shoulder[1]
    classification = "Left higher" if tilt < -20 else "Right higher" if tilt > 20 else "Level"
    return tilt, classification

def assess_kyphosis(points):
    ear = points[mp_pose.PoseLandmark.LEFT_EAR.value]
    shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]
    angle = get_angle(ear, shoulder, hip)
    classification = "Kyphosis" if angle < 160 else "Normal"
    return angle, classification

def assess_shoulder_elevation(points, frame):
    # Key landmarks
    shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    ear = points[mp_pose.PoseLandmark.LEFT_EAR.value]
    hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]

    # Calculate angle (ear–shoulder–hip)
    angle = get_angle(ear, shoulder, hip)

    # Classification
    classification = "Shoulder Elevated" if angle < 75 else "Normal"

    # # --- Overlay drawing ---
    # # Draw points
    # cv2.circle(frame, shoulder, 6, (0, 255, 0), -1)  # green
    # cv2.circle(frame, ear, 6, (255, 0, 0), -1)       # blue
    # cv2.circle(frame, hip, 6, (0, 0, 255), -1)       # red

    # # Draw lines
    # cv2.line(frame, shoulder, ear, (255, 255, 0), 2)
    # cv2.line(frame, shoulder, hip, (255, 255, 0), 2)

    # # Draw angle text
    # cv2.putText(frame, f"{int(angle)} deg", (shoulder[0] + 20, shoulder[1] - 20),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # # Classification text
    # cv2.putText(frame, classification, (shoulder[0] - 50, shoulder[1] - 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if classification == "Normal" else (0, 0, 255), 
    #             2, cv2.LINE_AA)

    return angle, classification

def assess_thoracic_tilt(points, frame):
    # Get landmarks
    left_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = points[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Midpoints (average of L/R)
    shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) // 2,
                    (left_shoulder[1] + right_shoulder[1]) // 2)
    hip_mid = ((left_hip[0] + right_hip[0]) // 2,
               (left_hip[1] + right_hip[1]) // 2)

    # Calculate tilt angle relative to vertical
    dx = shoulder_mid[0] - hip_mid[0]
    dy = hip_mid[1] - shoulder_mid[1]
    angle = math.degrees(math.atan2(dx, dy))
    tilt_angle = abs(angle)

    # Classification threshold
    classification = "Excessive Tilt" if tilt_angle > 20 else "Normal"

    # # --- Overlay drawing ---
    # # Draw midpoints
    # cv2.circle(frame, shoulder_mid, 6, (0, 255, 0), -1)
    # cv2.circle(frame, hip_mid, 6, (0, 0, 255), -1)

    # # Draw torso line
    # cv2.line(frame, shoulder_mid, hip_mid, (255, 255, 0), 2)

    # # Draw vertical reference line from hip upwards
    # cv2.line(frame, hip_mid, (hip_mid[0], hip_mid[1] - 200), (200, 200, 200), 2)

    # # Annotate angle + classification
    # cv2.putText(frame, f"{int(tilt_angle)} deg", (hip_mid[0] + 20, hip_mid[1] - 100),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # cv2.putText(frame, classification, (hip_mid[0] - 50, hip_mid[1] - 50),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0) if classification == "Normal" else (0, 0, 255),
    #             2, cv2.LINE_AA)

    return tilt_angle, classification

def assess_scapular_protraction(points):
    """
    Estimate scapular protraction based on horizontal displacement between
    shoulder and elbow (internal rotation forward).
    """
    shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    elbow = points[mp_pose.PoseLandmark.LEFT_ELBOW.value]

    # Horizontal displacement (positive = elbow in front of shoulder)
    displacement = elbow[0] - shoulder[0]

    # Classification thresholds (adjust based on your standards)
    if displacement > 20:
        classification = "Protracted"
    elif displacement < -20:
        classification = "Retracted"
    else:
        classification = "Normal"

    return displacement, classification

def assess_scapular_protraction_dynamic(points_sequence, frame_sequence):
    """
    Estimate scapular protraction dynamically over a movement sequence.
    points_sequence: list of frames, each frame is a list of landmarks [(x, y), ...]
    frame_sequence: corresponding frames (for overlay visualization)
    
    Returns:
        peak_displacement (float), classification (str)
    """
    displacements = []

    for points, frame in zip(points_sequence, frame_sequence):
        left_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = points[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = points[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Approximate torso midline x-coordinate
        torso_mid_x = (left_hip[0] + right_hip[0]) / 2

        # Shoulder displacement relative to torso midline
        left_disp = left_shoulder[0] - torso_mid_x
        right_disp = right_shoulder[0] - torso_mid_x

        # Take the larger forward displacement (dominant side)
        displacement = max(left_disp, right_disp)
        displacements.append(displacement)

        # --- Optional: overlay visualization ---
        cv2.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 6, (0, 255, 0), -1)
        cv2.circle(frame, (int(right_shoulder[0]), int(right_shoulder[1])), 6, (0, 255, 0), -1)
        cv2.line(frame, (int(torso_mid_x), 0), (int(torso_mid_x), frame.shape[0]), (255, 255, 0), 2)
        cv2.putText(frame, f"Disp: {int(displacement)} px", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Shoulder Posture Assessment', frame)
        cv2.waitKey(1)

    peak_disp = max(displacements)

    # Normalize by torso width for classification (optional)
    torso_width = abs(points_sequence[0][mp_pose.PoseLandmark.LEFT_SHOULDER.value][0] -
                      points_sequence[0][mp_pose.PoseLandmark.RIGHT_SHOULDER.value][0])
    norm_disp = peak_disp / torso_width * 100  # % of torso width

    # Classification thresholds (adjust to clinical standards)
    if norm_disp > 15:
        classification = "Protracted"
    elif norm_disp < -15:
        classification = "Retracted"
    else:
        classification = "Normal"

    return norm_disp, classification

def assess_subacromial_impingement(points_sequence, frame_sequence):
    """
    Estimate risk of sub-acromial impingement by measuring the vertical distance
    between acromion (shoulder landmark) and elbow (proxy for humeral head) during arm elevation.
    """
    distances = []

    for points, frame in zip(points_sequence, frame_sequence):
        left_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_elbow = points[mp_pose.PoseLandmark.LEFT_ELBOW.value]

        # Vertical distance (y-axis)
        distance = left_elbow[1] - left_shoulder[1]  # pixels
        distances.append(distance)

        # Overlay visualization
        cv2.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 6, (0, 255, 0), -1)
        cv2.circle(frame, (int(left_elbow[0]), int(left_elbow[1])), 6, (0, 0, 255), -1)
        cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                 (int(left_elbow[0]), int(left_elbow[1])), (255, 255, 0), 2)
        cv2.putText(frame, f"Dist: {int(distance)} px", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow('Shoulder Posture Assessment', frame)
        cv2.waitKey(1)

    min_distance = min(distances)

    # Normalize by torso height for classification (optional)
    torso_height = abs(points_sequence[0][mp_pose.PoseLandmark.LEFT_SHOULDER.value][1] -
                       points_sequence[0][mp_pose.PoseLandmark.LEFT_HIP.value][1])
    norm_distance = min_distance / torso_height * 100  # % of torso height

    classification = "Reduced Space (Risk)" if norm_distance < 20 else "Normal"

    return norm_distance, classification

def assess_bursitis_risk(points_sequence, frame_sequence):
    """
    Assess risk of bursitis by measuring scapular upward rotation and anterior tilt
    during arm elevation.
    """
    upward_rotations = []
    anterior_tilts = []

    for points, frame in zip(points_sequence, frame_sequence):
        left_shoulder = points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = points[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = points[mp_pose.PoseLandmark.RIGHT_HIP.value]

        # Torso midline
        torso_mid_x = (left_hip[0] + right_hip[0]) / 2
        torso_mid_y = (left_hip[1] + right_hip[1]) / 2

        # Upward rotation: angle between shoulder-hip line and vertical
        dx = left_shoulder[0] - torso_mid_x
        dy = torso_mid_y - left_shoulder[1]
        angle_upward = math.degrees(math.atan2(dx, dy))
        upward_rotations.append(angle_upward)

        # Anterior tilt: shoulder x relative to torso midline
        tilt = left_shoulder[0] - torso_mid_x
        anterior_tilts.append(tilt)

        # Overlay visualization
        cv2.circle(frame, (int(left_shoulder[0]), int(left_shoulder[1])), 6, (0, 255, 0), -1)
        cv2.circle(frame, (int(torso_mid_x), int(torso_mid_y)), 6, (255, 0, 0), -1)
        cv2.line(frame, (int(left_shoulder[0]), int(left_shoulder[1])),
                 (int(torso_mid_x), int(torso_mid_y)), (255, 255, 0), 2)
        cv2.putText(frame, f"UpRot: {int(angle_upward)} deg", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Tilt: {int(tilt)} px", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Shoulder Posture Assessment', frame)
        cv2.waitKey(1)

    # Peak values during movement
    max_upward_rotation = max(upward_rotations)
    max_anterior_tilt = max(anterior_tilts)

    # Classification thresholds (tunable)
    classification = "Poor Scapular Motion" if max_upward_rotation < 20 or max_anterior_tilt > 20 else "Normal"

    return max_upward_rotation, max_anterior_tilt, classification

def assess_nerve_compression(points_sequence, frame_sequence):
    """
    Assess nerve compression risk by combining:
    - Forward head posture (CVA) -> static measure
    - Rounded shoulders (elbow angle) -> static proxy
    - Dynamic scapular protraction -> internal rotation
    
    Returns:
        metrics: dict with all three metrics
        classification: overall risk classification
    """
    # --- Dynamic scapular protraction ---
    dyn_disp, dyn_class = assess_scapular_protraction_dynamic(points_sequence, frame_sequence)

    # --- Static Rounded Shoulder angle (average over first frame) ---
    first_points = points_sequence[0]
    left_shoulder = first_points[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow = first_points[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist = first_points[mp_pose.PoseLandmark.LEFT_WRIST.value]
    elbow_angle = get_angle(left_shoulder, left_elbow, left_wrist)
    elbow_class = "Rounded Shoulders" if elbow_angle < 150 else "Normal"

    # --- CVA (Forward Head) from first frame ---
    left_ear = first_points[mp_pose.PoseLandmark.LEFT_EAR.value]
    left_hip = first_points[mp_pose.PoseLandmark.LEFT_HIP.value]
    cva_angle = get_angle(left_ear, left_shoulder, left_hip)
    cva_class = "Forward Head" if cva_angle < 48 else "Normal"

    # --- Overall Classification ---
    # If any of the three indicate risk, classify as "At Risk"
    if dyn_class == "Protracted" or elbow_class == "Rounded Shoulders" or cva_class == "Forward Head":
        overall_class = "Nerve Compression Risk"
    else:
        overall_class = "Normal"

    metrics = {
        "CVA": cva_angle,
        "Elbow Angle": elbow_angle,
        "Scapular Protraction": dyn_disp
    }

    return metrics, overall_class

def capture_dynamic_sequence(cap, pose, capture_seconds=10, skip_frames=2):
    """
    Capture a dynamic movement sequence from the webcam for multiple dynamic modules.
    
    Args:
        cap: OpenCV VideoCapture object
        pose: MediaPipe Pose object
        capture_seconds: duration of capture in seconds (default 10)
        skip_frames: skip every Nth frame for visualization smoothing (default 2)
    
    Returns:
        points_sequence: list of landmark points per frame
        frame_sequence: list of corresponding frames
    """
    # Estimate FPS from webcam
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:  # fallback if FPS cannot be read
        fps = 30
    capture_frames = int(capture_seconds * fps)

    points_sequence = []
    frame_sequence = []

    # Countdown before capture
    ret, frame = cap.read()
    if not ret:
        return points_sequence, frame_sequence
    for i in range(3, 0, -1):
        countdown_frame = frame.copy()
        cv2.putText(countdown_frame, f"{i}", (frame.shape[1] // 2 - 20, frame.shape[0] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        cv2.imshow('Shoulder Posture Assessment', countdown_frame)
        cv2.waitKey(1000)

    # Capture loop
    for f in range(capture_frames):
        success, dyn_frame = cap.read()
        if not success:
            break
        dyn_frame = cv2.flip(dyn_frame, 1)
        rgb_dyn = cv2.cvtColor(dyn_frame, cv2.COLOR_BGR2RGB)
        results_dyn = pose.process(rgb_dyn)

        if results_dyn.pose_landmarks:
            landmarks = [(lm.x * dyn_frame.shape[1], lm.y * dyn_frame.shape[0])
                         for lm in results_dyn.pose_landmarks.landmark]
            points_sequence.append(landmarks)
            frame_sequence.append(dyn_frame)

            # Overlay only on every Nth frame to reduce CPU load
            if f % skip_frames == 0:
                cv2.putText(dyn_frame, f"Recording... {f//int(fps)+1}/{capture_seconds}s",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow('Shoulder Posture Assessment', dyn_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    return points_sequence, frame_sequence

def draw_multiline_text(frame, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale=0.6, color=(0, 255, 255), thickness=1, line_spacing=25, max_width=None):
    """
    Draw multi-line text on an OpenCV frame.
    Splits text into multiple lines if it exceeds max_width in pixels.
    
    Args:
        frame: the OpenCV frame
        text: string to draw
        pos: tuple (x, y) top-left corner
        font: OpenCV font
        font_scale: font size
        color: BGR tuple
        thickness: font thickness
        line_spacing: pixels between lines
        max_width: maximum width in pixels before wrapping (optional)
    """
    x, y = pos
    lines = [text]

    if max_width is not None:
        # Wrap text manually
        words = text.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + (' ' if current_line else '') + word
            (w, h), _ = cv2.getTextSize(test_line, font, font_scale, thickness)
            if w > max_width:
                if current_line:
                    lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        lines.append(current_line)

    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x, y + i * line_spacing), font, font_scale, color, thickness, cv2.LINE_AA)

# -----------------------------
# Flow Control
# -----------------------------
assessments = [
    ("CVA", assess_cva),
    ("Rounded Shoulders", assess_rounded_shoulders),
    ("Scapular Asymmetry", assess_scapular),
    ("Shoulder Tilt", assess_shoulder_tilt),
    ("Thoracic Kyphosis", assess_kyphosis),
    ("Shoulder Elevation Angle", assess_shoulder_elevation),
    ("Thoracic Tilt", assess_thoracic_tilt),
    ("Scapular Protraction", assess_scapular_protraction),
    ("Scapular Protraction (Dynamic)", assess_scapular_protraction_dynamic),
    ("Sub-acromial Impingement", assess_subacromial_impingement),
    ("Bursitis Risk (Scapular Rotation/Tilt)", assess_bursitis_risk),
    ("Nerve Compression Risk", assess_nerve_compression)
]

instructions = {
    "CVA": "Align sideways to camera. Press SPACE to start CVA test.",
    "Rounded Shoulders": "Face camera, arms relaxed. Press SPACE.",
    "Scapular Asymmetry": "Stand upright, face camera. Press SPACE.",
    "Shoulder Tilt": "Face camera, relax shoulders. Press SPACE.",
    "Thoracic Kyphosis": "Align sideways to camera. Press SPACE.",
    "Shoulder Elevation Angle": "Face camera, arms relaxed at side. Press SPACE.",
    "Thoracic Tilt": "Stand sideways to the camera, arms relaxed. Press SPACE.",
    "Scapular Protraction": "Face camera, relax arms. Press SPACE to assess scapular protraction.",
    "Scapular Protraction (Dynamic)": "Face camera, perform a small forward reaching movement. Press SPACE to start.",
    "Sub-acromial Impingement": "Raise your arm straight forward slowly. Press SPACE to start.",
    "Bursitis Risk (Scapular Rotation/Tilt)": "Raise your arm, allow natural scapular motion. Press SPACE to start.",
    "Nerve Compression Risk": "Face sideways to the camera, arms relaxed. "
    "Perform a small forward-reaching movement. Press SPACE to start."
}

# -----------------------------
# Main Loop
# -----------------------------
cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    assessment_index = 0
    
    while cap.isOpened() and assessment_index < len(assessments):
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            points = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.pose_landmarks.landmark]

            name, func = assessments[assessment_index]
            cv2.putText(frame, f"Assessment: {name}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, instructions[name], (20, 70),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            draw_multiline_text(frame, instructions[name], pos=(20, 70), font_scale=0.7,
                                color=(0, 255, 255), thickness=2, line_spacing=30, max_width=frame.shape[1] - 40)


        cv2.imshow('Shoulder Posture Assessment', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # SPACE pressed to start countdown
            show_countdown(frame)
        
            if results.pose_landmarks:
                points = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in results.pose_landmarks.landmark]
        
                # --- Dynamic Scapular Protraction Capture ---
                if name in ("Scapular Protraction (Dynamic)", "Sub-acromial Impingement", "Bursitis Risk (Scapular Rotation/Tilt)", "Nerve Compression Risk"):
                    # points_sequence = []
                    # frame_sequence = []
                    # capture_frames = 30  # ~1-2 seconds depending on webcam FPS
                    # for _ in range(capture_frames):
                    #     success, dyn_frame = cap.read()
                    #     if not success:
                    #         break
                    #     dyn_frame = cv2.flip(dyn_frame, 1)
                    #     rgb_dyn = cv2.cvtColor(dyn_frame, cv2.COLOR_BGR2RGB)
                    #     results_dyn = pose.process(rgb_dyn)
                    #     if results_dyn.pose_landmarks:
                    #         landmarks = [(lm.x * dyn_frame.shape[1], lm.y * dyn_frame.shape[0]) 
                    #                      for lm in results_dyn.pose_landmarks.landmark]
                    #         points_sequence.append(landmarks)
                    #         frame_sequence.append(dyn_frame)
        
                    # # Compute dynamic scapular protraction
                    # metric, classification = assess_scapular_protraction_dynamic(points_sequence, frame_sequence)
                    points_seq, frames_seq = capture_dynamic_sequence(cap, pose, capture_seconds=10, skip_frames=2)
                    if len(points_seq) == 0:
                        print("No landmarks detected. Retry the assessment.")
                        continue
                    result = func(points_seq, frames_seq)        
                else:
                    if func in (assess_shoulder_elevation, assess_thoracic_tilt):
                        metric, classification = func(points, frame)
                    else:
                        metric, classification = func(points)
        
                print(f"{name} → Metric: {metric:.2f}, Classification: {classification}")
                cv2.putText(frame, f"Result: {classification} ({metric:.2f})", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.putText(frame, "Press 'n' for next, 'r' to retry, 'q' to quit",
                            (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
                # Loop until user chooses
                decision_made = False
                while not decision_made:
                    cv2.imshow('Shoulder Posture Assessment', frame)
                    k = cv2.waitKey(100) & 0xFF
                    if k == ord('n'):
                        log_result(name, round(metric, 2), classification)
                        assessment_index += 1
                        decision_made = True
                    elif k == ord('r'):
                        decision_made = True
                    elif k == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()


        elif key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
