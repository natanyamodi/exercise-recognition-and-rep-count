# 🏋️ Exercise Classification and Rep Counting App
A video-based fitness analysis tool that classifies exercises and counts repetitions using pose landmarks extracted from uploaded videos. Built using Streamlit, MediaPipe, and a custom-trained classifier, it supports a variety of common exercises.

https://github.com/user-attachments/assets/5eb7217e-c770-421f-a132-69676d9d6345

## ✅Supported Exercises:
- Bicep Curl
- Squats
- Lunges
- Jumping Jacks
- Pushups

## 🌟 Overview
This application uses computer vision and a machine learning model to:
- Process uploaded workout videos
- Recognize the type of exercise being performed
- Track movement phases (e.g., "up/down" for squats, "arms up/down" for jumping jacks)
- Count repetitions with high accuracy (99% accuracy using Random Forest classifier)
- Display the live video and real-time results in a user-friendly Streamlit interface

## 🚀 Features
- Exercise Classification: Automatically identifies which exercise is being performed.
- Stage Detection: Determines movement phases to enable accurate rep counting
- Rep Counter: Tracks total reps per exercise
- Visual Feedback: Annotated frames showing current exercise, stage, angle, and rep summary in a table

## How It Works
- Pose Detection: Uses MediaPipe to track body landmarks
- Angle Calculation: Measures joint angles to determine movement stages
- Exercise Logic: Applies exercise-specific rules to count reps (pose_tracker.py)
- Prediction Model: A RandomForestClassifier trained on labeled pose landmark data
- Web App Interface: Built in Streamlit (app.py) for easy video upload and live feedback

## 🧪 Model Training
You can use the pre-trained model (exercises.pkl) or train your own using the provided structure.

### 📂 Dataset Structure for Training
```
exercises/
├── bicep_curl/       # Videos of bicep curls
├── squats/           # Videos of squats
├── lunges/           # Videos of lunges
├── jumping_jacks/    # Videos of jumping jacks
└── pushups/          # Videos of pushups
```
- Run the training notebook (exercise_classification.ipynb)
- The notebook will save the trained model as a .pkl file
- Place the model in the models folder

## 💻 Setup & Installation
To run the app locally, follow these steps: 

1. **Clone the Repository**
```
git clone https://github.com/natanyamodi/exercise-recognition-and-rep-count.git 
````

2. **Create Virtual Environment**
```
python -m venv venv
``` 
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

3. **Install Dependencies**
```
pip install -r requirements.txt
```
  
5. **Run the App**
```
streamlit run app.py
```
