import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Enhances skin detection using color thresholding and morphological operations.
    
    - Converts the frame to HSV and YCrCb color spaces.
    - Applies inRange() to extract skin-like regions.
    - Uses morphological operations to remove noise.
    
    Returns:
        skin_mask: Binary mask highlighting skin areas.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color ranges in YCrCb and HSV color spaces
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    lower_hsv = np.array([0, 40, 50], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    
    # Create masks based on color ranges
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    # Combine both masks using bitwise AND
    skin_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    
    return skin_mask

def detect_hand(frame):
    """
    Detects the hand, palm center, fingertips, knuckles, and wrist points using contours and convex hull.

    - Extracts the largest skin-like contour.
    - Uses convex hull and convexity defects to identify fingertips and knuckles.
    - Calculates the palm center using image moments.
    - Identifies the lowest hull points as wrist points.

    Returns:
        frame: The frame with detected hand landmarks drawn.
    """
    processed_mask = preprocess_frame(frame)
    
    # Find contours in the processed mask
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame  # Return original frame if no contours found
    
    # Get the largest contour (assumed to be the hand)
    max_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(max_contour) < 5000:
        return frame  # Ignore small contours (likely noise)
    
    # Compute the convex hull and convexity defects
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)
    hull_points = cv2.convexHull(max_contour, returnPoints=True)
    
    # Draw the hand contour
    cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
    
    # Compute the palm center using image moments
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
        palm_center = (cx, cy)
        cv2.circle(frame, palm_center, 10, (255, 255, 0), -1)  # Draw palm center (Cyan)
    else:
        palm_center = None
    
    # Detect fingertips and knuckles
    fingers = []
    knuckles = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            
            # Calculate angles to distinguish fingertips from knuckles
            a = np.linalg.norm(np.array(start) - np.array(far))
            b = np.linalg.norm(np.array(end) - np.array(far))
            c = np.linalg.norm(np.array(start) - np.array(end))
            angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b)) * 57  # Convert to degrees
            
            if angle < 80 and d > 5000:
                fingers.append(start)
                fingers.append(end)
            elif angle > 80:
                knuckles.append(far)
    
    # Filter unique fingertips and knuckles
    unique_fingers = []
    unique_knuckles = []
    for f in fingers:
        if all(np.linalg.norm(np.array(f) - np.array(uf)) > 20 for uf in unique_fingers):
            unique_fingers.append(f)
    for k in knuckles:
        if all(np.linalg.norm(np.array(k) - np.array(uk)) > 20 for uk in unique_knuckles):
            unique_knuckles.append(k)
    
    # Identify wrist points (lowest hull points)
    hull_points = sorted(hull_points, key=lambda x: x[0][1], reverse=True)
    wrist_points = hull_points[:2] if len(hull_points) > 1 else []
    
    # Draw fingertips, knuckles, and wrist points
    for fingertip in unique_fingers:
        cv2.circle(frame, fingertip, 7, (0, 0, 255), -1)  # Draw fingertips (Red)
        if palm_center:
            cv2.line(frame, palm_center, fingertip, (255, 255, 255), 2)
    
    for knuckle in unique_knuckles:
        cv2.circle(frame, knuckle, 7, (0, 255, 255), -1)  # Draw knuckles (Yellow)
    
    for wrist in wrist_points:
        cv2.circle(frame, tuple(wrist[0]), 7, (255, 0, 255), -1)  # Draw wrist points (Magenta)
    
    return frame

# Start real-time video capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply hand detection
    frame = detect_hand(frame)
    
    # Display result
    cv2.imshow("Enhanced Hand Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
