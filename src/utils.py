import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from google import genai
from google.genai import types

from IPython.display import clear_output, Image, display
from PIL import Image as PILImage
import io


def display_jupyter(processed):
    """Display frame(s) in Jupyter notebook"""
    
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    pil_img = PILImage.fromarray(processed_rgb)

    # Create binary stream
    bio = io.BytesIO()
    pil_img.save(bio, format="PNG")

    display(Image(data=bio.getvalue()))
    clear_output(wait=True)  # Clear previous frame


def display_cv2(processed):
    """Display frame(s) using OpenCV window"""
    cv2.imshow("Video Feed", processed)
    
def create_video_saver(filename, frame_size, fps=30, codec='H264'):
    """Creates a VideoWriter using the provided filename, fps, and codec,
    and returns a function that writes frames to the video file.
    """
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
    
    def save_frame(frame_to_save):
        writer.write(frame_to_save)
    
    # Attaching the release method to the save_frame function to enable cleanup
    save_frame.release = writer.release
    
    return save_frame


def setup_gesture_recognizer():
    """Initialize and return the MediaPipe gesture recognizer"""
    base_options = python.BaseOptions(model_asset_path="models/gesture_recognizer.task")
    options = vision.GestureRecognizerOptions(base_options=base_options)
    return vision.GestureRecognizer.create_from_options(options)


def detect_keypoint(frame, gesture_recognizer):
    """Process frame to detect pointing gesture and get index fingertip"""
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
                int(hand_landmarks[8].y * frame.shape[0]),
            )
            return index_tip

    return None


def detect_closed_fist(frame, gesture_recognizer):
    """Process frame to detect closed fist gesture"""
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    gesture_result = gesture_recognizer.recognize(mp_image)

    if gesture_result.gestures and gesture_result.hand_landmarks:
        top_gesture = gesture_result.gestures[0]
        # Check for Closed_Fist gesture
        if top_gesture[0].category_name == "Closed_Fist":
            return True

    return False


def setup_text_detector(api_key):
    """Initialize and return the GenAI client"""
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Failed to initialize recognizer: {e}")
        return None


def detect_text(client, image):
    """Process image to detect text using GenAI"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            "What is written in this image? Only answer with the text. If there is no text, return an empty string.",
            image,
        ],
    )

    return response.text

def detect_text_from_canvas(canvas, text_detector):
    """Process canvas and detect text"""
   
    # Prepare canvas for text detection
    canvas_for_ocr = cv2.bitwise_not(canvas)
    gray_canvas = cv2.cvtColor(canvas_for_ocr, cv2.COLOR_BGR2GRAY)
    _, binary_canvas = cv2.threshold(gray_canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    padded_canvas = cv2.copyMakeBorder(binary_canvas, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    # Save the canvas (optional)
    cv2.imwrite('canvas_output.png', canvas)
    
    pil_image = PILImage.fromarray(padded_canvas)
    
    try:
        text = detect_text(text_detector, pil_image)
        if text.strip():
            return text.strip()
    except Exception as e:
        print(f"Text Detection Error: {e}")
        return None
    
def draw_recognized_text(frame, text):
    """Draw recognized text as subtitle on frame"""
    if text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 30
        
        # Draw black background
        cv2.rectangle(frame, 
                    (text_x - 10, text_y - text_size[1] - 10),
                    (text_x + text_size[0] + 10, text_y + 10),
                    (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
