# Video Tracking Project

## MUSI Image and Video Analysis

This repository contains the project files, code, and notebooks for the Video Tracking project under the Image and Video Analysis course in the Masters of Intelligent Systems (MUSI) at the University of the Balearic Islands.

### Team

- Umar Faruk Abdullahi
- Raixa Madueño

## Introduction

In the field of Human-Computer Interaction (HCI), the quest for more intuitive and seamless methods of computer control has been ongoing for a long time. Our project presents an approach that relies on direct camera input to enable handwriting through real-time finger tracking. By using advances in computer vision and machine learning, we capture the motions of a user’s finger and convert these movements into words.

## Approach

To solve this problem, we rely on object tracking. Object tracking is a computer vision task that involves locating and following one or more objects over time within a video sequence. The primary goal is to determine the trajectory of an object as it moves through scenes, maintaining its identity as it moves across the frames. To achieve this, we follow the steps below:

1. **Detection:** The first task is to correctly identify the desired object of interest to track. In our case, the `tip of index finger`. This is achieved using the [MediaPipe library](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#get_started) which is an open-source tool that allows hand tracking and gesture recognition.
2. **Tracking:** After correctly detecting the point of interest, we use a tracking algorithm to follow the point of interest as it moves across frames thereby accumulating the trajectory. Two algorithms were used in the project:
   1. Kalman Filters
   2. Mean Shift
3. **Correction:** As the finger moves across frames, we correct our tracking error by providing the correct location of the finger at the subsequent frame. This can be directly incorporated within the tracking algorithm or made additional object detection. In our project, we rely on the tracking algorithm.
4. **Handwriting Recognition:** After every subsequent word is written, we run the captured frame through an optical character recognition (OCR) model to extract the written text. We use the [Google Gemini Model](https://ai.google.dev/gemini-api/docs/vision?lang=python) for the OCR.

The full step-by-step tutorial is provided in the project [Notebook](./Hand-Tracking-Pipeline.ipynb)

## Setup

### Prerequisites

- Python 3.7+
- OpenCV
- NumPy
- MediaPipe
- Google GenAI

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/Lidnmp/tracking_video.git
    cd tracking_video
    ```

2. Create and activate a virtual environment:

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Set up the environment variables:
    - Create a `.env` file in the root directory of the project.
    - Add your Google Gemini API key:

        ```
        GEMINI_API_KEY=<your_key>
        ```

## Usage

1. Run the Jupyter Notebook:

    ```sh
    jupyter notebook [Hand-Tracking-Pipeline.ipynb](http://_vscodecontentref_/1)
    ```

2. Follow the steps in the notebook to initialize the gesture recognizer, set up the Kalman filter, and start the tracking process.

## Code Structure

- `Hand-Tracking-Pipeline.ipynb`: Main notebook containing the implementation of the tracking pipeline.
- `src/utils.py`: Utility functions for gesture recognition, text detection, and display.
- `src/kalman_filter.py`: Implementation of the Kalman Filter for tracking.


## Acknowledgements

- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#get_started)
- [Google Gemini Model](https://ai.google.dev/gemini-api/docs/vision?lang=python)
- [Machine Learning Space](https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/)