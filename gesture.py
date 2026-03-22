import cv2
import numpy as np
from ultralytics import YOLO
import math

# --- Load YOLO Pose Model ---
model = YOLO("yolov8n-pose.pt")

# --- Helper Functions ---
def distance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def detect_gesture(person, frame_shape):
    """Detect gestures from a single person's keypoints."""
    required_indices = [0, 5, 6, 9, 10]  # nose, shoulders, wrists
    
    # Check if required keypoints are present and confident
    if any(i >= len(person) or person[i][2] < 0.3 for i in required_indices):
        return "Incomplete keypoints"

    # Get keypoints
    nose = person[0][:2]
    left_shoulder = person[5][:2]
    right_shoulder = person[6][:2]
    left_wrist = person[9][:2]
    right_wrist = person[10][:2]

    # Convert to pixel coordinates for better distance calculation
    h, w = frame_shape[:2]
    
    nose_px = (int(nose[0] * w), int(nose[1] * h))
    left_shoulder_px = (int(left_shoulder[0] * w), int(left_shoulder[1] * h))
    right_shoulder_px = (int(right_shoulder[0] * w), int(right_shoulder[1] * h))
    left_wrist_px = (int(left_wrist[0] * w), int(left_wrist[1] * h))
    right_wrist_px = (int(right_wrist[0] * w), int(right_wrist[1] * h))

    # Calculate distances in pixels
    head_radius = distance(nose_px, left_shoulder_px) * 0.8  # Approximate head radius
    chest_region_height = distance(left_shoulder_px, (left_shoulder_px[0], left_shoulder_px[1] + head_radius))

    # Debug: Print distances
    # print(f"Head radius: {head_radius:.1f}px, Chest height: {chest_region_height:.1f}px")

    # 1. Hand on Head Detection
    left_wrist_to_nose = distance(left_wrist_px, nose_px)
    right_wrist_to_nose = distance(right_wrist_px, nose_px)
    
    if left_wrist_to_nose < head_radius or right_wrist_to_nose < head_radius:
        return "🤦 Hand on Head"

    # 2. Hand on Chest Detection
    # Define chest region (between shoulders and slightly below)
    chest_center_y = (left_shoulder_px[1] + right_shoulder_px[1]) / 2
    chest_region_top = chest_center_y - chest_region_height * 0.2
    chest_region_bottom = chest_center_y + chest_region_height * 0.8
    
    # Check if wrists are in chest region vertically
    left_wrist_in_chest = chest_region_top < left_wrist_px[1] < chest_region_bottom
    right_wrist_in_chest = chest_region_top < right_wrist_px[1] < chest_region_bottom
    
    # Check if wrists are near the chest horizontally (between shoulders)
    chest_left_x = min(left_shoulder_px[0], right_shoulder_px[0])
    chest_right_x = max(left_shoulder_px[0], right_shoulder_px[0])
    
    left_wrist_near_chest = chest_left_x - 50 < left_wrist_px[0] < chest_right_x + 50
    right_wrist_near_chest = chest_left_x - 50 < right_wrist_px[0] < chest_right_x + 50
    
    if (left_wrist_in_chest and left_wrist_near_chest) or (right_wrist_in_chest and right_wrist_near_chest):
        return "🫀 Hand on Chest"

    # 3. Hands Raised Detection
    if left_wrist_px[1] < nose_px[1] and right_wrist_px[1] < nose_px[1]:
        return "🙌 Both Hands Raised"
    elif left_wrist_px[1] < nose_px[1]:
        return "✋ Left Hand Raised"
    elif right_wrist_px[1] < nose_px[1]:
        return "✋ Right Hand Raised"

    # 4. Crossed Arms Detection
    left_wrist_to_right_shoulder = distance(left_wrist_px, right_shoulder_px)
    right_wrist_to_left_shoulder = distance(right_wrist_px, left_shoulder_px)
    
    if left_wrist_to_right_shoulder < head_radius and right_wrist_to_left_shoulder < head_radius:
        return "❌ Crossed Arms"

    return "No Emergency Gesture"

# --- Main Loop ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected or feed empty")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, verbose=False)

    gesture_texts = []
    keypoints_detected = None

    for res in results:
        if hasattr(res, "keypoints") and res.keypoints is not None:
            keypoints = res.keypoints.data.cpu().numpy()
            keypoints_detected = keypoints
            
            # Process each detected person
            for i, person in enumerate(keypoints):
                gesture = detect_gesture(person, frame.shape)
                gesture_texts.append(f"Person {i+1}: {gesture}")
                
                # Draw keypoints and connections for better visualization
                for kp in person:
                    if len(kp) >= 3 and kp[2] > 0.3:
                        x, y = int(kp[0] * frame.shape[1]), int(kp[1] * frame.shape[0])
                        cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
                
                # Draw skeleton connections for better understanding
                connections = [
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
                    (5, 11), (6, 12), (11, 12),  # Body
                    (0, 1), (0, 2), (1, 3), (2, 4)  # Face (simplified)
                ]
                
                for start, end in connections:
                    if (start < len(person) and end < len(person) and 
                        person[start][2] > 0.3 and person[end][2] > 0.3):
                        start_pt = (int(person[start][0] * frame.shape[1]), 
                                  int(person[start][1] * frame.shape[0]))
                        end_pt = (int(person[end][0] * frame.shape[1]), 
                                int(person[end][1] * frame.shape[0]))
                        cv2.line(frame, start_pt, end_pt, (0, 255, 0), 2)

    # Display gesture texts
    if gesture_texts:
        for i, text in enumerate(gesture_texts):
            color = (0, 255, 0) if any(g in text for g in ["Hand", "Raised", "Crossed"]) else (0, 0, 255)
            cv2.putText(frame, text, (10, 30 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(frame, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Emergency Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()