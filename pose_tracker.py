import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict


def calculate_angle(a: List[float], b: List[float], c: List[float]) -> float:
    """Calculate the angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    return 360 - angle if angle > 180.0 else angle


class PoseAnalyzer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        
        # Maintain separate counters and stages for each exercise
        self.exercise_counters = {
            "squats": 0,
            "bicep_curl": 0,
            "lunges": 0,
            "jumping_jacks": 0,
            "pushups": 0
        }
        
        self.exercise_stages = {
            "squats": None,
            "bicep_curl": None,
            "lunges": None,
            "jumping_jacks": None,
            "pushups": None
        }
        
        # Exercise-specific parameters
        self.exercise_params = {
            "bicep_curl": {
                "joints": ["shoulder", "elbow", "wrist"],
                "thresholds": {"up": 160, "down": 30}
            },
            "squats": {
                "joints": ["hip", "knee", "ankle"],
                "thresholds": {"up": 160, "down": 90}
            },
            "lunges": {
                "joints": ["hip", "knee", "ankle"],
                "thresholds": {"up": 160, "down": 90}
            },
            "pushups": {
                "joints": ["shoulder", "elbow", "wrist"],
                "thresholds": {"up": 160, "down": 90}
            },
            "jumping_jacks": {
                "joints": ["hip", "shoulder", "wrist"],
                "thresholds": {"up": 160, "down": 20}
            }
        }

    def get_landmark_coordinates(self, results, joint_name: str) -> Optional[Tuple[float, float]]:
        """Get coordinates for a specific joint from pose landmarks"""
        landmarks = {
            "shoulder": [self.mp_pose.PoseLandmark.LEFT_SHOULDER, self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
            "elbow": [self.mp_pose.PoseLandmark.LEFT_ELBOW, self.mp_pose.PoseLandmark.RIGHT_ELBOW],
            "wrist": [self.mp_pose.PoseLandmark.LEFT_WRIST, self.mp_pose.PoseLandmark.RIGHT_WRIST],
            "hip": [self.mp_pose.PoseLandmark.LEFT_HIP, self.mp_pose.PoseLandmark.RIGHT_HIP],
            "knee": [self.mp_pose.PoseLandmark.LEFT_KNEE, self.mp_pose.PoseLandmark.RIGHT_KNEE],
            "ankle": [self.mp_pose.PoseLandmark.LEFT_ANKLE, self.mp_pose.PoseLandmark.RIGHT_ANKLE]
        }
        
        if joint_name not in landmarks:
            return None
            
        # Try left side first, then right if left not visible
        for landmark in landmarks[joint_name]:
            landmark_data = results.pose_landmarks.landmark[landmark]
            if landmark_data.visibility > 0.5:
                return (landmark_data.x, landmark_data.y)
        return None

    def analyze_exercise(self, image, results, exercise: str) -> Dict:
        """Analyze the current frame for the specified exercise"""
        if exercise not in self.exercise_params:
            return {
                "counter": self.exercise_counters.get(exercise, 0),
                "stage": self.exercise_stages.get(exercise, None),
                "angle": 0,
                "joint_position": None
            }
            
        params = self.exercise_params[exercise]
        joints = params["joints"]
        thresholds = params["thresholds"]
        
        # Get joint coordinates
        a = self.get_landmark_coordinates(results, joints[0])
        b = self.get_landmark_coordinates(results, joints[1])
        c = self.get_landmark_coordinates(results, joints[2])
        
        if not all([a, b, c]):
            return {
                "counter": self.exercise_counters[exercise],
                "stage": self.exercise_stages[exercise],
                "angle": 0,
                "joint_position": None
            }
            
        # Calculate angle
        angle = calculate_angle(a, b, c)
        
        # Get current counter and stage for this exercise
        counter = self.exercise_counters[exercise]
        stage = self.exercise_stages[exercise]
        
        # Update counter based on exercise logic
        if exercise == "bicep_curl":
            if angle < thresholds["down"]:
                stage = "up"
            if angle > thresholds["up"] and stage == "up":
                stage = "down"
                counter += 1
                
        elif exercise == "squats" or exercise == "lunges":
            if angle < thresholds["down"]:
                stage = "down"
            if angle > thresholds["up"] and stage == "down":
                stage = "up"
                counter += 1
                
        elif exercise == "pushups":
            if angle > thresholds["up"]:
                stage = "up"
            if angle < thresholds["down"] and stage == "up":
                stage = "down"
                counter += 1
                
        elif exercise == "jumping_jacks":
            if angle < thresholds["down"]:
                stage = "arms down"
            if angle > thresholds["up"] and stage == "arms down":
                stage = "arms up"
                counter += 1
        
        # Update the exercise's counter and stage
        self.exercise_counters[exercise] = counter
        self.exercise_stages[exercise] = stage
        
        return {
            "counter": counter,
            "stage": stage,
            "angle": angle,
            "joint_position": b  # Position to display the angle (middle joint)
        }

    def get_all_counters(self) -> Dict[str, int]:
        """Return all exercise counters"""
        return self.exercise_counters.copy()