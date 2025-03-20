# HAND-LANDMARKS-DETECTION
<img src="https://github.com/user-attachments/assets/ab5da0a4-d42c-4b73-addb-2902149bb2ac" width="500">
<img src="https://github.com/user-attachments/assets/1592485b-be46-414f-ab57-8e190faaa0b9" width="500">



## Overview
This project implements a computer vision-based hand detection system using OpenCV. The program processes live video input to detect the hand, fingertips, knuckles, palm center, and wrist points using contour and convex hull analysis.

## APPROACH:
  This project detects hand landmarks using only computer vision algorithms without relying on models like MediaPipe.

## Features
- **Skin Detection**: Uses color thresholding in HSV and YCrCb color spaces.
- **Hand Contour Extraction**: Identifies the largest hand-like region.
- **Convex Hull & Convexity Defects**: Detects fingertips, knuckles, and wrist points.
- **Palm Center Calculation**: Uses image moments to find the centroid of the hand.
- **Real-time Video Processing**: Captures and processes video frames from a webcam.

## Computer Vision Techniques Used
1. **Color Segmentation**:
   - Thresholding in the HSV and YCrCb color spaces to isolate skin regions.
   - Morphological operations (opening and closing) to remove noise.
2. **Contour Analysis**:
   - `cv2.findContours()` is used to detect the hand region.
   - `cv2.moments()` computes the center of mass for the hand.
3. **Convex Hull and Convexity Defects**:
   - `cv2.convexHull()` finds the boundary around the hand.
   - `cv2.convexityDefects()` identifies fingertip valleys to determine finger and knuckle positions.
4. **Wrist Detection**:
   - The lowest points on the convex hull are selected as wrist points.

## Usage Instructions
1. Ensure you have OpenCV (`cv2`) and NumPy installed.
2. Run the script to start real-time hand detection.
3. Press `q` to exit the program.



