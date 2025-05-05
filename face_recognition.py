import cv2
import os
import dlib
import numpy as np
import math
from collections import deque
from download_models import main as download_predictor

def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Initialize dlib's face detector and facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.isfile(predictor_path):
        print(f"Predictor file not found. Downloading...")
        download_predictor()
    
    try:
        # Path to facial landmarks predictor file
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)
    except Exception as e:
        print(f"Error loading face predictor: {e}")
        print("Please download the file from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        return
    
    # Create window
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Face Aligned", cv2.WINDOW_NORMAL)
    
    # Initialize variables for stabilization
    history_length = 10
    position_history = deque(maxlen=history_length)
    angle_history = deque(maxlen=history_length)
    scale_history = deque(maxlen=history_length)
    
    # Smoothing factors (higher = smoother but less responsive)
    position_smoothing = 0.9
    angle_smoothing = 0.85 
    scale_smoothing = 0.9
    
    # Variables to track previous values
    prev_face_center = None
    prev_angle = 0
    prev_scale = 1.0
    face_detected_prev = False
    
    # Target face width
    target_width = 250
    
    # Default crop dimensions
    crop_width = int(target_width * 1.5)
    crop_height = int(crop_width * 1.2)
    
    while True:
        # Capture frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Create a copy of the original frame to draw on
        display_frame = frame.copy()
        
        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        face_detected = len(faces) > 0
        
        # Process the largest face if any faces are detected
        if face_detected:
            # Find the largest face
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get facial landmarks
            landmarks = predictor(gray, largest_face)
            
            # Get coordinates for drawing
            x, y, w, h = largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()
            
            # Draw rectangle around the face
            cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Get eye coordinates for alignment
            left_eye_x = (landmarks.part(36).x + landmarks.part(39).x) // 2
            left_eye_y = (landmarks.part(36).y + landmarks.part(39).y) // 2
            right_eye_x = (landmarks.part(42).x + landmarks.part(45).x) // 2
            right_eye_y = (landmarks.part(42).y + landmarks.part(45).y) // 2
            
            # Draw circles at eye centers
            cv2.circle(display_frame, (left_eye_x, left_eye_y), 3, (255, 0, 0), -1)
            cv2.circle(display_frame, (right_eye_x, right_eye_y), 3, (255, 0, 0), -1)
            
            # Calculate angle for alignment
            dx = right_eye_x - left_eye_x
            dy = right_eye_y - left_eye_y
            current_angle = math.degrees(math.atan2(dy, dx))
            
            # Calculate eye center for rotation
            eye_center = ((left_eye_x + right_eye_x) // 2, (left_eye_y + right_eye_y) // 2)
            
            # Determine the scale to maintain consistent face size
            current_scale = target_width / w
            
            # Stabilize parameters with smoothing
            if not face_detected_prev:
                # First face detection, initialize values
                stabilized_center = eye_center
                stabilized_angle = current_angle
                stabilized_scale = current_scale
                
                # Initialize history
                position_history.extend([eye_center] * history_length)
                angle_history.extend([current_angle] * history_length)
                scale_history.extend([current_scale] * history_length)
            else:
                # Add current values to history
                position_history.append(eye_center)
                angle_history.append(current_angle)
                scale_history.append(current_scale)
                
                # Apply temporal smoothing with exponential moving average
                if prev_face_center is not None:
                    stabilized_center = (
                        int(prev_face_center[0] * position_smoothing + eye_center[0] * (1 - position_smoothing)),
                        int(prev_face_center[1] * position_smoothing + eye_center[1] * (1 - position_smoothing))
                    )
                    
                    # Handle angle wrapping (transition between -180 and 180)
                    angle_diff = current_angle - prev_angle
                    if angle_diff > 180:
                        angle_diff -= 360
                    elif angle_diff < -180:
                        angle_diff += 360
                    
                    stabilized_angle = prev_angle + angle_diff * (1 - angle_smoothing)
                    stabilized_scale = prev_scale * scale_smoothing + current_scale * (1 - scale_smoothing)
                else:
                    stabilized_center = eye_center
                    stabilized_angle = current_angle
                    stabilized_scale = current_scale
            
            # Create rotation matrix with stabilized parameters
            rot_mat = cv2.getRotationMatrix2D(stabilized_center, stabilized_angle, stabilized_scale)
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Apply affine transformation to align and scale the face
            aligned_face = cv2.warpAffine(frame, rot_mat, (width, height), flags=cv2.INTER_CUBIC)
            
            # Calculate the face center in the aligned image
            face_center_x = int(rot_mat[0][0] * stabilized_center[0] + rot_mat[0][1] * stabilized_center[1] + rot_mat[0][2])
            face_center_y = int(rot_mat[1][0] * stabilized_center[0] + rot_mat[1][1] * stabilized_center[1] + rot_mat[1][2])
            
            # Calculate crop boundaries
            crop_x1 = max(0, face_center_x - crop_width // 2)
            crop_y1 = max(0, face_center_y - crop_height // 2)
            crop_x2 = min(width, crop_x1 + crop_width)
            crop_y2 = min(height, crop_y1 + crop_height)
            
            # Perform the crop
            face_cropped = aligned_face[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Show the cropped face if valid
            if face_cropped.size > 0:
                cv2.imshow("Face Aligned", face_cropped)
            
            # Update previous values
            prev_face_center = stabilized_center
            prev_angle = stabilized_angle
            prev_scale = stabilized_scale
        elif face_detected_prev and prev_face_center is not None:
            # Face lost but we have previous values - continue showing last valid frame
            # (helps prevent flickering when face detection momentarily fails)
            pass
        
        # Remember if face was detected this frame
        face_detected_prev = face_detected
        
        # Show the original frame with face detection
        cv2.imshow("Original", display_frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()