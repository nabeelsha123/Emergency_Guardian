"""
GUARDIAN NET - FALL DETECTION SYSTEM
Complete working version with custom model support
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import warnings
import os
import sys

# Add the current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from guardian_integration import GuardianAlertSender

warnings.filterwarnings('ignore')

class GuardianFallDetector:
    def __init__(self, patient_id=1):
        print("\n" + "="*70)
        print("🚀 GUARDIAN NET - FALL DETECTION SYSTEM")
        print("="*70)
        
        # Get paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.project_root = os.path.dirname(self.script_dir)
        
        # Model paths
        self.custom_model_path = os.path.join(
            self.project_root, 'runs', 'train', 'fall_detection', 'weights', 'best.pt'
        )
        self.pretrained_model_path = os.path.join(self.script_dir, 'yolov8n-pose.pt')
        
        # Load model
        self.load_model()
        
        # Guardian Net integration
        self.patient_id = patient_id
        self.alert_sender = GuardianAlertSender(patient_id=patient_id)
        
        # Test connection
        if self.alert_sender.test_connection():
            print("✅ Connected to Guardian Net server")
            print(f"📱 Patient ID: {patient_id}")
        else:
            print("⚠️ Cannot connect to server - alerts logged locally")
        
        # Detection parameters
        self.state = "MONITORING"
        self.total_falls = 0
        self.consecutive_fall_frames = 0
        self.consecutive_stand_frames = 0
        self.fall_start_time = 0
        
        self.fall_confidence_threshold = 0.65
        self.required_fall_frames = 5
        self.required_stand_frames = 8
        
        self.fall_confidence_history = deque(maxlen=8)
        self.pose_history = deque(maxlen=10)
        
        print("✅ Fall detection system ready!")
        print("="*70 + "\n")
    
    def load_model(self):
        """Load the best available model"""
        print("\n🔍 Loading model...")
        
        # Try custom trained model
        if os.path.exists(self.custom_model_path):
            print(f"✅ Found custom trained model!")
            print(f"   Path: {self.custom_model_path}")
            try:
                self.pose_model = YOLO(self.custom_model_path)
                self.model_type = "CUSTOM TRAINED ✓"
                self.model_color = (0, 255, 0)
                print("   ✅ Using custom trained fall detection model")
                
                # Try to get model info
                try:
                    import yaml
                    dataset_path = os.path.join(self.project_root, 'dataset', 'human-fall-detection', 'data.yaml')
                    if os.path.exists(dataset_path):
                        with open(dataset_path, 'r') as f:
                            data = yaml.safe_load(f)
                            print(f"   Classes: {data.get('names', ['fall', 'normal'])}")
                except:
                    pass
                    
            except Exception as e:
                print(f"   ❌ Error loading custom model: {e}")
                self.load_pretrained()
        else:
            self.load_pretrained()
    
    def load_pretrained(self):
        """Load pretrained model as fallback"""
        print(f"⚠️ Custom model not found")
        print(f"   Expected: {self.custom_model_path}")
        
        if os.path.exists(self.pretrained_model_path):
            print(f"   Loading pretrained model...")
            self.pose_model = YOLO(self.pretrained_model_path)
            self.model_type = "PRETRAINED (YOLOv8-pose)"
            self.model_color = (200, 200, 200)
            print("   ⚠️ Using pretrained YOLOv8-pose model")
            print("   💡 Train custom model for better accuracy:")
            print("      python detector/train_final.py")
        else:
            print(f"   ❌ No model found!")
            sys.exit(1)
    
    def calculate_fall_confidence(self, keypoints, frame_shape):
        """Calculate confidence that a fall occurred"""
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        
        confidence_scores = []
        keypoints = keypoints[0]
        
        # Check body angle
        if len(keypoints) >= 13:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            if (left_shoulder[2] > 0.2 and right_shoulder[2] > 0.2 and 
                left_hip[2] > 0.2 and right_hip[2] > 0.2):
                
                shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
                hip_center = (left_hip[:2] + right_hip[:2]) / 2
                
                dx = hip_center[0] - shoulder_center[0]
                dy = hip_center[1] - shoulder_center[1]
                
                angle = np.degrees(np.arctan2(abs(dx), abs(dy))) if abs(dy) > 0.001 else 90.0
                angle_confidence = max(0.0, min(1.0, (angle - 30) / 60.0))
                confidence_scores.append(angle_confidence * 0.5)
        
        # Check aspect ratio
        if len(keypoints) >= 17:
            valid_points = [kp for kp in keypoints if kp[2] > 0.2]
            if len(valid_points) >= 4:
                y_coords = [kp[1] for kp in valid_points]
                x_coords = [kp[0] for kp in valid_points]
                
                height = max(y_coords) - min(y_coords)
                width = max(x_coords) - min(x_coords)
                
                if width > 0 and height > 0:
                    aspect_ratio = height / width
                    if aspect_ratio < 1.0:
                        aspect_confidence = 1.0
                    elif aspect_ratio < 2.0:
                        aspect_confidence = 1.5 - (aspect_ratio / 2.0)
                    else:
                        aspect_confidence = 0.0
                    confidence_scores.append(aspect_confidence * 0.3)
        
        # Check vertical position
        if len(keypoints) >= 17:
            ankle_indices = [15, 16]
            valid_ankles = [keypoints[i] for i in ankle_indices if keypoints[i][2] > 0.2]
            
            if valid_ankles:
                ankle_y = max([kp[1] for kp in valid_ankles])
                ground_confidence = min(1.0, ankle_y * 1.5)
                confidence_scores.append(ground_confidence * 0.2)
        
        return min(1.0, sum(confidence_scores)) if confidence_scores else 0.0
    
    def calculate_stand_confidence(self, keypoints):
        """Calculate confidence that person is standing"""
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        
        stand_scores = []
        keypoints = keypoints[0]
        
        if len(keypoints) >= 13:
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            
            if (left_shoulder[2] > 0.2 and right_shoulder[2] > 0.2 and 
                left_hip[2] > 0.2 and right_hip[2] > 0.2):
                
                shoulder_center = (left_shoulder[:2] + right_shoulder[:2]) / 2
                hip_center = (left_hip[:2] + right_hip[:2]) / 2
                
                dx = hip_center[0] - shoulder_center[0]
                dy = hip_center[1] - shoulder_center[1]
                
                angle = np.degrees(np.arctan2(abs(dx), abs(dy))) if abs(dy) > 0.001 else 90.0
                
                if angle < 25:
                    stand_confidence = 1.0
                elif angle < 45:
                    stand_confidence = 1.0 - ((angle - 25) / 20.0)
                else:
                    stand_confidence = 0.0
                
                stand_scores.append(stand_confidence)
        
        if len(keypoints) >= 17:
            valid_points = [kp for kp in keypoints if kp[2] > 0.2]
            if len(valid_points) >= 4:
                height = max([kp[1] for kp in valid_points]) - min([kp[1] for kp in valid_points])
                stand_scores.append(min(1.0, height * 2.0) * 0.5)
        
        return np.mean(stand_scores) if stand_scores else 0.0
    
    def update_state_machine(self, fall_confidence, stand_confidence):
        """Update fall detection state machine"""
        current_time = time.time()
        
        if self.state == "MONITORING":
            if fall_confidence > self.fall_confidence_threshold:
                self.consecutive_fall_frames += 1
                self.fall_confidence_history.append(fall_confidence)
                
                if (self.consecutive_fall_frames >= self.required_fall_frames and 
                    np.mean(self.fall_confidence_history) > self.fall_confidence_threshold):
                    
                    self.state = "FALL_DETECTED"
                    self.fall_start_time = current_time
                    self.total_falls += 1
                    self.consecutive_stand_frames = 0
                    
                    # Send alert
                    message = f"🚨 Fall detected with {fall_confidence:.1%} confidence!"
                    self.alert_sender.send_alert("fall", message, fall_confidence)
                    
                    print("\n" + "!"*50)
                    print(f"🚨 FALL DETECTED! (Confidence: {fall_confidence:.2f})")
                    print(f"   Total falls: {self.total_falls}")
                    print("!"*50 + "\n")
            else:
                self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 2)
        
        elif self.state == "FALL_DETECTED":
            if stand_confidence > 0.7:
                self.consecutive_stand_frames += 1
                if self.consecutive_stand_frames >= self.required_stand_frames:
                    self.state = "MONITORING"
                    self.consecutive_fall_frames = 0
                    self.fall_confidence_history.clear()
                    print("✅ Person stood up - monitoring resumed")
            elif current_time - self.fall_start_time > 30:
                self.state = "MONITORING"
                self.consecutive_fall_frames = 0
                self.fall_confidence_history.clear()
    
    def process_frame(self, frame):
        """Process a single frame"""
        processing_frame = cv2.resize(frame, (640, 480))
        results = self.pose_model(processing_frame, verbose=False, conf=0.5, imgsz=320)
        
        fall_confidence = 0.0
        stand_confidence = 0.0
        keypoints = None
        
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            if len(keypoints) > 0:
                fall_confidence = self.calculate_fall_confidence(keypoints, frame.shape)
                stand_confidence = self.calculate_stand_confidence(keypoints)
        
        self.update_state_machine(fall_confidence, stand_confidence)
        return fall_confidence, stand_confidence, keypoints
    
    def draw_results(self, frame, fall_confidence, keypoints):
        """Draw detection results on frame"""
        color = (0, 255, 0) if self.state == "MONITORING" else (0, 0, 255)
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (400, 170), (0, 0, 0), -1)
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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.model_color, 1)
        
        # Confidence bar
        bar_x, bar_y = 15, 120
        bar_w, bar_h = 200, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * fall_confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"Fall: {fall_confidence:.2f}", (bar_x + bar_w + 10, bar_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Fall count
        cv2.putText(frame, f"Total Falls: {self.total_falls}", (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if self.total_falls > 0 else (255, 255, 255), 1)
        
        # Draw keypoints
        if keypoints is not None and len(keypoints) > 0:
            keypoints = keypoints[0]
            h, w = frame.shape[:2]
            
            for kp in keypoints:
                if kp[2] > 0.2:
                    x, y = int(kp[0] * w / 640), int(kp[1] * h / 480)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        return frame

def main():
    # Configuration
    PATIENT_ID = 1
    
    print("\n" + "-"*50)
    print(f"Guardian Net - Fall Detection")
    print(f"Patient ID: {PATIENT_ID}")
    print("-"*50 + "\n")
    
    # Initialize detector
    detector = GuardianFallDetector(patient_id=PATIENT_ID)
    
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
    
    # FPS calculation
    fps_start = time.time()
    fps_count = 0
    fps = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to grab frame")
            break
        
        # Process frame
        fall_conf, stand_conf, keypoints = detector.process_frame(frame)
        
        # Draw results
        frame = detector.draw_results(frame, fall_conf, keypoints)
        
        # Update FPS
        fps_count += 1
        if time.time() - fps_start >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_start = time.time()
        
        cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Show frame
        cv2.imshow("Guardian Net - Fall Detection", frame)
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.total_falls = 0
            print("📊 Fall counter reset")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\n📊 Session Summary:")
    print(f"   Falls detected: {detector.total_falls}")
    print(f"   Alerts sent: {detector.alert_sender.alert_count}")
    print(f"   Model used: {detector.model_type}")
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()