# Exercise Classification and Rep Counting App
A video-based fitness tracking tool that uses computer vision to analyze uploaded workout videos, classify exercises, and count reps using body landmarks. Powered by MediaPipe and Streamlit, it supports multiple exercises like squats, pushups, lunges, bicep curls and jumping jacks.

https://github.com/user-attachments/assets/5eb7217e-c770-421f-a132-69676d9d6345


## 🌟 Overview
This application uses computer vision to:
- Detect body movements in real-time from video input
- Track exercise stages (e.g., "up/down" for squats, "arms up/down" for jumping jacks)
- Count repetitions for supported exercises
- Display results in a clean table format

## 🚀 Features
- Stage Detection: Identifies movement phases (e.g., "up", "down", "arms up")
- Rep Counting: Automatically counts repetitions for each exercise
- Multi-Exercise Support: Works with squats, bicep curls, lunges, jumping jacks, and pushups

## How It Works
- Pose Detection: Uses MediaPipe to track body landmarks
- Angle Calculation: Measures joint angles to determine movement stages
- Exercise Logic: Applies exercise-specific rules to count reps
- Table Display: Shows real-time results in a Streamlit table

## 🧪 Model Training
The exercise classifier was trained on a dataset of pose landmarks from various exercises. You have two options:

**Option 1:** Use Pre-trained Model
The model is already included in the models folder as exercises.pkl.

**Option 2:** Train Your Own Model

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
1. Clone the Repository
```
git clone https://github.com/natanyamodi/exercise-recognition-and-rep-count.git 
````

2. Create Virtual Environment
```
python -m venv venv
``` 
Windows: venv\Scripts\activate
Mac/Linux: source venv/bin/activate

3. Install Dependencies
```
pip install -r requirements.txt
```
  
5. Run the App
```
streamlit run app.py
```
