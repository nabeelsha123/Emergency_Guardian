"""
GUARDIAN NET - FALL DETECTION USING OBJECT DETECTION
Works with datasets that detect 'fallen', 'sitting', 'standing'
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import warnings
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from guardian_integration import GuardianAlertSender

warnings.filterwarnings('ignore')

class GuardianFallDetectorObjectDetection:
    def __init__(self, patient_id=1):
        print("\n" + "="*70)
        print("🚀 GUARDIAN NET - FALL DETECTION (OBJECT DETECTION)")
        print("="*70)
        
        # Get paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Load custom object detection model
        custom_model_path = os.path.join(self.project_root, 'runs', 'train', 'fall_detection', 'weights', 'best.pt')
        
        if os.path.exists(custom_model_path):
            print(f"✅ Found custom model: {custom_model_path}")
            self.model = YOLO(custom_model_path)
            self.model_type = "CUSTOM (Object Detection)"
            
            # Get class names
            import yaml
            dataset_path = os.path.join(self.project_root, 'dataset', 'human-fall-detection', 'data.yaml')
            if os.path.exists(dataset_path):
                with open(dataset_path, 'r') as f:
                    data = yaml.safe_load(f)
                    self.class_names = data.get('names', ['fallen', 'sitting', 'standing'])
                    print(f"   Classes: {self.class_names}")
        else:
            print(f"❌ Model not found: {custom_model_path}")
            print("   Please train first using: python detector/train_final.py")
            sys.exit(1)
        
        # Guardian Net integration
        self.patient_id = patient_id
        self.alert_sender = GuardianAlertSender(patient_id=patient_id)
        
        if self.alert_sender.test_connection():
            print("✅ Connected to Guardian Net server")
        else:
            print("⚠️ Cannot connect to server")
        
        # Detection parameters
        self.state = "MONITORING"
        self.total_falls = 0
        self.fall_detection_history = deque(maxlen=10)
        self.fall_confidence_threshold = 0.5
        self.required_frames = 3  # Need 3 consecutive detections
        
        print("✅ Fall detection system ready!")
        print("="*70 + "\n")
    
    def process_frame(self, frame):
        """Process frame with object detection model"""
        results = self.model(frame, verbose=False, conf=0.5)
        
        fall_detected = False
        fall_confidence = 0.0
        boxes = []
        
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Check if this is a fall (class 0 is usually 'fallen')
                if cls == 0 and conf > self.fall_confidence_threshold:
                    fall_detected = True
                    fall_confidence = max(fall_confidence, conf)
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    boxes.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        # Update state machine
        self.update_state_machine(fall_detected, fall_confidence)
        
        return fall_detected, fall_confidence, boxes
    
    def update_state_machine(self, fall_detected, confidence):
        """Update state based on detections"""
        self.fall_detection_history.append(fall_detected)
        
        # Check if we have enough consecutive fall detections
        if len(self.fall_detection_history) >= self.required_frames:
            recent_detections = list(self.fall_detection_history)[-self.required_frames:]
            
            if all(recent_detections):
                if self.state == "MONITORING":
                    self.state = "FALL_DETECTED"
                    self.total_falls += 1
                    
                    # Send alert
                    message = f"🚨 Fall detected with {confidence:.1%} confidence!"
                    self.alert_sender.send_alert("fall", message, confidence)
                    
                    print("\n" + "!"*50)
                    print(f"🚨 FALL DETECTED! (Confidence: {confidence:.2f})")
                    print(f"   Total falls: {self.total_falls}")
                    print("!"*50 + "\n")
            else:
                if self.state == "FALL_DETECTED":
                    self.state = "MONITORING"
                    print("✅ Resuming monitoring")
    
    def draw_results(self, frame, fall_detected, confidence, boxes):
        """Draw detection results"""
        color = (0, 0, 255) if self.state == "FALL_DETECTED" else (0, 255, 0)
        
        # Draw info panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # State
        cv2.putText(frame, f"STATE: {self.state}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Patient info
        cv2.putText(frame, f"Patient ID: {self.patient_id}", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Alerts: {self.alert_sender.alert_count}", (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Model info
        cv2.putText(frame, f"Model: {self.model_type}", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Confidence bar
        bar_x, bar_y = 15, 120
        bar_w, bar_h = 200, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"Conf: {confidence:.2f}", (bar_x + bar_w + 10, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Fall count
        cv2.putText(frame, f"Falls: {self.total_falls}", (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if self.total_falls > 0 else (255, 255, 255), 1)
        
        # Draw bounding boxes
        for x1, y1, x2, y2, conf in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"FALLEN {conf:.2f}" if self.state == "FALL_DETECTED" else f"Person {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame

def main():
    PATIENT_ID = 1
    
    print("\n" + "-"*50)
    print(f"Guardian Net - Object Detection Fall Detection")
    print(f"Patient ID: {PATIENT_ID}")
    print("-"*50 + "\n")
    
    detector = GuardianFallDetectorObjectDetection(patient_id=PATIENT_ID)
    
    # Open camera
    cap = None
    for camera_index in [0, 1, 2]:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                print(f"✅ Camera found at index {camera_index}")
                break
            cap.release()
            cap = None
    
    if cap is None or not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n🎥 Camera started. Press 'q' to quit, 'r' to reset counter\n")
    
    fps_start = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break
        
        # Process frame
        fall_detected, confidence, boxes = detector.process_frame(frame)
        
        # Draw results
        frame = detector.draw_results(frame, fall_detected, confidence, boxes)
        
        # Update FPS
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_start = time.time()
        
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Guardian Net - Fall Detection", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.total_falls = 0
            print("📊 Fall counter reset")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 Session Summary:")
    print(f"   Falls detected: {detector.total_falls}")
    print(f"   Alerts sent: {detector.alert_sender.alert_count}")
    print(f"   Model used: {detector.model_type}")
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()