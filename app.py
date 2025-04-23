import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import tempfile
import os
import time
from pose_tracker import PoseAnalyzer

# Path to the trained model
MODEL_PATH = 'models/exercises.pkl'

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Load the trained model with Streamlit caching
@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    return model

# Load the model
model = load_model()

# Initialize PoseAnalyzer
pose_analyzer = PoseAnalyzer()

def display_exercise_info(image, current_exercise, confidence, exercise_counters, 
                        exercise_stages, angle, joint_position, width, height):
    """Display exercise information on the image"""
    # Dynamically adjust font sizes
    font_scale = max(width, height) / 1000
    font_thickness = 1
    padding = int(min(width, height) * 0.02)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Left panel - Current exercise info
    current_text = f"Current: {current_exercise}"
    confidence_text = f"Confidence: {confidence:.0%}"
    stage_text = f"Stage: {exercise_stages[current_exercise]}" if exercise_stages[current_exercise] else "Stage: None"
    
    # Calculate left panel size
    (current_w, current_h), _ = cv2.getTextSize(current_text, font, font_scale, font_thickness)
    (conf_w, conf_h), _ = cv2.getTextSize(confidence_text, font, font_scale, font_thickness)
    (stage_w, stage_h), _ = cv2.getTextSize(stage_text, font, font_scale, font_thickness)
    
    left_width = max(current_w, conf_w, stage_w) + 2 * padding
    left_height = current_h + conf_h + stage_h + 4 * padding
    
    # Draw left panel
    cv2.rectangle(image, (10, 10), (10 + left_width, 10 + left_height), (255, 255, 255), -1)
    cv2.putText(image, current_text, (10 + padding, 10 + padding + current_h), 
               font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(image, confidence_text, (10 + padding, 10 + 2 * padding + current_h + conf_h), 
               font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(image, stage_text, (10 + padding, 10 + 3 * padding + current_h + conf_h + stage_h), 
               font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    
    # Display angle if available
    if joint_position and angle > 0:
        joint_x = int(joint_position[0] * width)
        joint_y = int(joint_position[1] * height)
        angle_text = f"{int(angle)}"
        (angle_w, angle_h), _ = cv2.getTextSize(angle_text, font, font_scale, font_thickness)
        cv2.putText(image, angle_text, 
                   (joint_x - angle_w//2, joint_y + angle_h//2), 
                   font, font_scale, (255, 255, 255), font_thickness+1, cv2.LINE_AA)

# Settings
CONFIDENCE_THRESHOLD = 0.5
MIN_CONSECUTIVE_FRAMES = 5
HISTORY_SIZE = 5
EXERCISES = ["squats", "bicep_curl", "lunges", "jumping_jacks", "pushups"]

# Streamlit app
st.title("Exercise Classification and Rep Counting App")
st.write("Upload a video to classify the exercise and count reps")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Reset counters for new video
    pose_analyzer = PoseAnalyzer()
    
    # Create temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    # Initialize video capture
    cap = cv2.VideoCapture(tfile.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create placeholders
    video_placeholder = st.empty()
    summary_placeholder = st.empty()
    status_text = st.empty()
    
    # Initialize tracking variables
    prediction_history = []
    current_exercise = None
    confidence = 0
    consecutive_frames = 0
    
    # Process video and display
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            
            if results.pose_landmarks:
                # Draw landmarks and process exercise
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )
                
                # Predict exercise
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                      for landmark in results.pose_landmarks.landmark]).flatten())
                X = pd.DataFrame([pose_row])
                exercise_prob = model.predict_proba(X)[0]
                max_prob = np.max(exercise_prob)
                predicted_exercise = model.predict(X)[0]
                
                # Update prediction history
                if max_prob > CONFIDENCE_THRESHOLD:
                    prediction_history.append(predicted_exercise)
                    if len(prediction_history) > HISTORY_SIZE:
                        prediction_history.pop(0)
                    
                    new_exercise = max(set(prediction_history), key=prediction_history.count)
                    
                    if new_exercise != current_exercise:
                        consecutive_frames += 1
                        if consecutive_frames >= MIN_CONSECUTIVE_FRAMES:
                            current_exercise = new_exercise
                            consecutive_frames = 0
                    else:
                        consecutive_frames = 0
                    
                    confidence = max_prob
                
                if current_exercise:
                    analysis = pose_analyzer.analyze_exercise(image, results, current_exercise)
                    display_exercise_info(image, current_exercise, confidence, 
                                        pose_analyzer.exercise_counters,
                                        pose_analyzer.exercise_stages,
                                        analysis['angle'], analysis['joint_position'],
                                        width, height)
            
            # Convert to JPEG for better streaming performance
            _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            frame_bytes = buffer.tobytes()
            
            # Display video and summary
            video_placeholder.image(frame_bytes, channels="BGR", use_container_width=True)
            
            # Update summary
            summary_data = []
            for ex in EXERCISES:
                label = f"➡️ {ex}" if ex == current_exercise else ex
                summary_data.append((label, pose_analyzer.exercise_counters[ex]))
            
            summary_df = pd.DataFrame(summary_data, columns=["Exercise", "Repetitions"])
            summary_placeholder.dataframe(summary_df, use_container_width=True, height=250)
            
            # Control playback speed
            processing_time = time.time() - start_time
            delay = max(1/fps - processing_time, 0)
            time.sleep(delay)
    
    # Cleanup
    cap.release()
    
    cv2.destroyAllWindows()
    