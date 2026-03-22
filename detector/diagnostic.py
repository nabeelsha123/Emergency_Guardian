"""
DIAGNOSTIC SCRIPT - Check what your model detects
"""

import cv2
from ultralytics import YOLO
import os

print("\n" + "="*70)
print("🔍 MODEL DIAGNOSTIC")
print("="*70)

# Find the model
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

model_paths = [
    os.path.join(project_root, 'runs', 'train', 'fall_custom_scratch', 'weights', 'best.pt'),
    os.path.join(project_root, 'runs', 'train', 'fall_detection', 'weights', 'best.pt'),
]

model = None
for path in model_paths:
    if os.path.exists(path):
        print(f"\n✅ Found model: {path}")
        model = YOLO(path)
        break

if not model:
    print("❌ No model found!")
    exit()

# Open camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("\n🎥 Camera opened. Showing all detections...")
print("   This will show what your model actually sees")
print("   Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection with low confidence to see everything
    results = model(frame, verbose=False, conf=0.2)
    
    # Create display frame
    display = frame.copy()
    
    # Draw all detections
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        print(f"\rDetections: {len(boxes)}", end="")
        
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            
            # Class names (based on your dataset)
            class_names = ['fallen', 'sitting', 'standing']
            class_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
            
            # Color based on class
            if cls == 0:  # fallen
                color = (0, 0, 255)  # Red
            elif cls == 1:  # sitting
                color = (0, 255, 255)  # Yellow
            else:  # standing
                color = (0, 255, 0)  # Green
            
            # Draw box
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(display, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    # Show info
    cv2.putText(display, "DIAGNOSTIC MODE - Showing all detections", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(display, "Press 'q' to quit", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Model Diagnostic - What does your model detect?", display)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n\n" + "="*70)
print("DIAGNOSTIC COMPLETE")
print("="*70)
print("\nBased on what you saw:")
print("- If you saw GREEN boxes (standing) - model detects standing people")
print("- If you saw YELLOW boxes (sitting) - model detects sitting")
print("- If you saw RED boxes (fallen) - model detects falls")
print("- If you saw NO boxes - model is not detecting anything")
print("\nTell me what you saw in the camera feed!")