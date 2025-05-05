# METU Artificial Intelligence Society Face Recognition Project

A multi-phase face recognition system built with OpenCV and dlib.

## Project Roadmap

This project is being developed in three sprints:

1. **Sprint 1 (Current)**: Face detection, alignment, and stabilization
2. **Sprint 2 (Planned)**: Face recognition and identification
3. **Sprint 3 (Planned)**: Emotion detection

## Features (Sprint 1)

- Real-time face detection using dlib
- Face alignment based on eye positions
- Temporal smoothing for stable face tracking
- Consistent face size through dynamic scaling
- Cropped view of aligned face

## Requirements

- Python 3.6+
- OpenCV
- dlib
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/YOUR_USERNAME/METU-AI-FaceRecognition.git
   cd METU-AI-FaceRecognition
   ```

2. Install required packages:
   ```
   pip install opencv-python dlib numpy
   ```

3. Run the application:
   ```
   python face_recognition.py
   ```
   
   *Note: The application will automatically download the required shape predictor file (shape_predictor_68_face_landmarks.dat) on first run if it's not found.*

## Usage

- Two windows will appear:
  - "Original": Shows the webcam feed with face detection
  - "Face Aligned": Shows the aligned and cropped face
- Press 'q' to exit the application

## Project Structure

```
METU-AI-FaceRecognition/
├── face_recognition.py   # Main script for face detection and alignment
├── download_models.py    # Helper script to download required model files
└── .gitignore            # Specifies files to exclude from Git
```

## Future Development

### Sprint 2: Face Recognition and Identification
- Implement face recognition using facial embeddings
- Create a database of known faces
- Display person's name when recognized

### Sprint 3: Emotion Detection
- Detect and classify facial emotions
- Display emotion labels on recognized faces
- Create emotion statistics

## Troubleshooting

- If you encounter "Error: Could not open camera", make sure your webcam is properly connected and not being used by another application.
- If you get "Error loading face predictor", ensure your internet connection is working for automatic download of the shape predictor file.
