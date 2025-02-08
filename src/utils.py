import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from IPython.display import clear_output, Image, display
from PIL import Image as PILImage
import io

def display_jupyter(frame, processed=None):
    """Display frame(s) in Jupyter notebook"""
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display side by side if we have two frames
    if processed is not None:
        
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        combined = np.hstack((rgb_frame, processed_rgb))
        pil_img = PILImage.fromarray(combined)
    else:
        pil_img = PILImage.fromarray(rgb_frame)
    
    # Create binary stream
    bio = io.BytesIO()
    pil_img.save(bio, format='PNG')
    
    display(Image(data=bio.getvalue()))
    clear_output(wait=True)  # Clear previous frame
    
def display_cv2(frame, processed=None):
    """Display frame(s) using OpenCV window"""
    if processed is not None:
        # Stack frames horizontally
        combined = np.hstack((frame, processed))
        cv2.imshow('Video Feed', combined)
    else:
        cv2.imshow('Video Feed', frame)
        
def detect_initial_point(frame):
    """Process frame to detect pointing gesture and get index fingertip"""
    
    # Initialize gesture recognizer of mediapipe
    base_options = python.BaseOptions(model_asset_path='models/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
    
    # Initialize index fingertip to None
    index_tip = None
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    
    gesture_result = gesture_recognizer.recognize(mp_image)
    
    # The landmark 8 is the tip of the index finger based on the mediapipe hand landmarks
    # Available at https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#get_started
    if gesture_result.gestures and gesture_result.hand_landmarks:
        top_gesture = gesture_result.gestures[0]
        if top_gesture[0].category_name == "Pointing_Up":
            hand_landmarks = gesture_result.hand_landmarks[0]
            index_tip = (
                int(hand_landmarks[8].x * frame.shape[1]),
                int(hand_landmarks[8].y * frame.shape[0])
            )
            gesture_recognizer.close()
            return index_tip
    
    gesture_recognizer.close()
    return None
