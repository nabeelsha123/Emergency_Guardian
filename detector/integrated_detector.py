#!/usr/bin/env python
"""
GUARDIAN NET - INTEGRATED FALL & VOICE DETECTOR
Combines high-accuracy fall detection with multilingual voice emergency detection
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import threading
import speech_recognition as sr
import requests
import winsound
import json
from datetime import datetime
import queue
import warnings

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
SERVER_URL = "http://localhost:3000"
PATIENT_ID = 1  # Will be updated based on database

# Emergency keywords in multiple languages
EMERGENCY_KEYWORDS = {
    'english': [
        'help', 'emergency', 'accident', 'fall', 'fell', 'fallen',
        'hurt', 'pain', 'injured', 'bleeding', 'help me',
        'save me', 'ambulance', 'doctor', 'hospital', 'fire',
        'thief', 'danger', 'urgent', 'come fast', 'need help'
    ],
    'malayalam': [
        'സഹായം', 'അടിയന്തരം', 'അപകടം', 'വീഴ്ച', 'വീണു',
        'വേദന', 'രക്തസ്രാവം', 'പരിക്ക്', 'വേഗം വരൂ',
        'ആംബുലൻസ്', 'ആശുപത്രി', 'ഡോക്ടർ', 'തീ', 'കള്ളൻ'
    ],
    'hindi': [
        'मदद', 'आपातकाल', 'दुर्घटना', 'गिर गया', 'चोट',
        'दर्द', 'जल्दी आओ', 'खतरा', 'बचाओ', 'एम्बुलेंस'
    ]
}

# ==================== ALERT SENDER ====================
class AlertSender:
    def __init__(self, patient_id=1):
        self.server_url = SERVER_URL
        self.patient_id = patient_id
        self.alert_endpoint = f"{SERVER_URL}/api/detector/alert"
        self.status_endpoint = f"{SERVER_URL}/api/detector/status-update"
        self.last_alert_time = 0
        self.alert_cooldown = 8  # seconds
        self.alert_count = 0
        self.alert_queue = queue.Queue()
        self.running = True
        
        # Start alert worker thread
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        
    def _worker(self):
        """Background thread to send alerts"""
        while self.running:
            try:
                alert = self.alert_queue.get(timeout=1)
                self._send_alert(alert)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Alert worker error: {e}")
    
    def _send_alert(self, alert):
        """Actually send the alert"""
        try:
            response = requests.post(self.alert_endpoint, json=alert, timeout=2)
            if response.status_code == 200:
                self.alert_count += 1
                print(f"\n✅ ALERT SENT! (#{self.alert_count})")
            else:
                print(f"\n❌ Server error: {response.status_code}")
        except Exception as e:
            print(f"\n❌ Connection error: {e}")
    
    def send_alert(self, alert_type, message, confidence=None, keywords=None):
        """Queue an alert to be sent"""
        current_time = time.time()
        
        # Cooldown check
        if current_time - self.last_alert_time < self.alert_cooldown:
            return False
        
        self.last_alert_time = current_time
        
        # Prepare payload
        payload = {
            "patient_id": self.patient_id,
            "alert_type": alert_type,
            "message": message,
            "confidence": float(confidence) if confidence else None,
            "keywords": keywords
        }
        
        # Remove None values
        payload = {k: v for k, v in payload.items() if v is not None}
        
        # Queue for sending
        self.alert_queue.put(payload)
        return True
    
    def update_status(self, state=None, fall_active=None, voice_active=None):
        """Update detector status on server"""
        try:
            payload = {}
            if state: payload['state'] = state
            if fall_active is not None: payload['fall_active'] = fall_active
            if voice_active is not None: payload['voice_active'] = voice_active
            
            requests.post(self.status_endpoint, json=payload, timeout=1)
        except:
            pass
    
    def stop(self):
        self.running = False

# ==================== FALL DETECTOR ====================
class FallDetector:
    def __init__(self, alert_sender):
        print("   📹 Initializing Fall Detection...")
        self.alert_sender = alert_sender
        
        # Load YOLO model
        try:
            self.pose_model = YOLO('yolov8n-pose.pt')
            print("   ✅ YOLO model loaded")
        except Exception as e:
            print(f"   ❌ Failed to load YOLO model: {e}")
            print("   Please run: pip install ultralytics")
            sys.exit(1)
        
        self.state = "MONITORING"
        self.total_falls = 0
        self.consecutive_fall_frames = 0
        self.consecutive_stand_frames = 0
        self.fall_confidence_history = deque(maxlen=8)
        self.fall_confidence_threshold = 0.65
        self.required_fall_frames = 5
        self.required_stand_frames = 8
        self.last_alert_time = 0
        self.running = True
        
        print("   ✅ Fall Detection Ready")

    def calculate_fall_confidence(self, keypoints):
        """Calculate confidence that a fall has occurred"""
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        
        confidence_scores = []
        keypoints = keypoints[0]
        
        # Check body angle (shoulder-hip line)
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
        
        # Check aspect ratio (width vs height)
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
        
        # Check vertical position (near ground)
        if len(keypoints) >= 17:
            ankle_indices = [15, 16]
            valid_ankles = [keypoints[i] for i in ankle_indices if keypoints[i][2] > 0.2]
            
            if valid_ankles:
                ankle_y = max([kp[1] for kp in valid_ankles])
                ground_confidence = min(1.0, ankle_y * 1.5)
                confidence_scores.append(ground_confidence * 0.2)
        
        return float(min(1.0, sum(confidence_scores))) if confidence_scores else 0.0

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
        
        return float(np.mean(stand_scores)) if stand_scores else 0.0

    def update_state_machine(self, fall_confidence, stand_confidence):
        """Update FSM based on confidences"""
        if self.state == "MONITORING":
            if fall_confidence > self.fall_confidence_threshold:
                self.consecutive_fall_frames += 1
                self.fall_confidence_history.append(fall_confidence)
                
                if (self.consecutive_fall_frames >= self.required_fall_frames and 
                    np.mean(self.fall_confidence_history) > self.fall_confidence_threshold):
                    
                    self.state = "FALL_DETECTED"
                    self.total_falls += 1
                    self.consecutive_stand_frames = 0
                    
                    message = f"Fall detected with {fall_confidence:.1%} confidence"
                    print(f"\n🔥 FALL DETECTED! Confidence: {fall_confidence:.2f}")
                    
                    # Play alert sound
                    for _ in range(3):
                        winsound.Beep(1000, 200)
                        time.sleep(0.1)
                    
                    # Send alert
                    self.alert_sender.send_alert("fall", message, fall_confidence)
                    self.alert_sender.update_status(state="FALL_DETECTED")
            else:
                self.consecutive_fall_frames = max(0, self.consecutive_fall_frames - 2)
        
        elif self.state == "FALL_DETECTED":
            if stand_confidence > 0.7:
                self.consecutive_stand_frames += 1
                if self.consecutive_stand_frames >= self.required_stand_frames:
                    self.state = "MONITORING"
                    self.consecutive_fall_frames = 0
                    self.fall_confidence_history.clear()
                    print("   ✅ Person stood up")
                    self.alert_sender.update_status(state="MONITORING")

    def process_frame(self, frame):
        """Process a single frame for fall detection"""
        processing_frame = cv2.resize(frame, (640, 480))
        results = self.pose_model(processing_frame, verbose=False, conf=0.5, imgsz=320)
        
        fall_confidence = 0.0
        stand_confidence = 0.0
        keypoints = None
        
        if results and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()
            if len(keypoints) > 0:
                fall_confidence = self.calculate_fall_confidence(keypoints)
                stand_confidence = self.calculate_stand_confidence(keypoints)
        
        self.update_state_machine(fall_confidence, stand_confidence)
        return fall_confidence, stand_confidence, keypoints

    def draw_results(self, frame, fall_confidence):
        """Draw detection results on frame"""
        # Set color based on state
        if self.state == "FALL_DETECTED":
            color = (0, 0, 255)  # Red
            bg_color = (0, 0, 200)
        else:
            color = (0, 255, 0)  # Green
            bg_color = (0, 200, 0)
        
        # Draw background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        cv2.putText(frame, f"STATE: {self.state}", (15, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"FALL CONF: {fall_confidence:.2f}", (15, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"TOTAL FALLS: {self.total_falls}", (15, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw progress bar for fall confidence
        bar_x, bar_y = 15, 100
        bar_w, bar_h = 200, 15
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
        fill_w = int(bar_w * min(fall_confidence, 1.0))
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
        cv2.putText(frame, f"{int(fall_confidence*100)}%", (bar_x + bar_w + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def run(self):
        """Main fall detection loop"""
        # Try to open camera
        cap = None
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"   ✅ Fall camera found at index {i}")
                break
        
        if cap is None or not cap.isOpened():
            print("   ❌ Cannot open fall detection camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("   📹 Fall detection camera started")
        self.alert_sender.update_status(fall_active=True)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            fall_conf, _, _ = self.process_frame(frame)
            frame = self.draw_results(frame, fall_conf)
            
            cv2.imshow("Guardian Net - Fall Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('r'):
                self.total_falls = 0
                print("   📊 Reset fall counter")

        cap.release()
        cv2.destroyAllWindows()
        self.alert_sender.update_status(fall_active=False)

    def stop(self):
        self.running = False

# ==================== VOICE DETECTOR ====================
class VoiceDetector:
    def __init__(self, alert_sender):
        print("   🎤 Initializing Voice Detection...")
        self.alert_sender = alert_sender
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.emergency_count = 0
        self.keywords = EMERGENCY_KEYWORDS
        self.supported_languages = ['en-IN', 'ml-IN', 'hi-IN', 'en-US']
        self.last_alert_time = 0
        self.running = True
        self.audio_queue = queue.Queue()
        
        # Calibrate microphone
        print("   🔊 Calibrating microphone...")
        with self.microphone as source:
            try:
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print("   ✅ Microphone calibrated")
            except Exception as e:
                print(f"   ⚠️ Calibration error: {e}")
        
        # Start audio processor thread
        self.processor_thread = threading.Thread(target=self._process_audio_queue, daemon=True)
        self.processor_thread.start()
        
        print("   ✅ Voice Detection Ready")

    def _process_audio_queue(self):
        """Background thread to process audio"""
        while self.running:
            try:
                audio = self.audio_queue.get(timeout=1)
                self._process_audio(audio)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Audio processor error: {e}")

    def check_keywords(self, text):
        """Check if text contains emergency keywords"""
        text_lower = text.lower()
        found = []
        for lang, words in self.keywords.items():
            for word in words:
                if word.lower() in text_lower:
                    found.append(word)
        return list(dict.fromkeys(found))

    def _process_audio(self, audio):
        """Process captured audio for keywords"""
        try:
            # Try different languages
            text = None
            lang_used = None
            
            for lang in self.supported_languages:
                try:
                    text = self.recognizer.recognize_google(audio, language=lang)
                    lang_used = lang
                    break
                except:
                    continue
            
            if text:
                timestamp = datetime.now().strftime('%H:%M:%S')
                print(f"   🗣️ [{timestamp}] Heard: '{text}'")
                
                keywords = self.check_keywords(text)
                
                if keywords and (time.time() - self.last_alert_time) > 10:
                    self.last_alert_time = time.time()
                    self.emergency_count += 1
                    message = f"Emergency detected! Keywords: {', '.join(keywords)}"
                    print(f"\n🔊 VOICE EMERGENCY DETECTED! Keywords: {', '.join(keywords)}")
                    
                    # Play alert sound
                    for freq in [800, 1000, 1200]:
                        winsound.Beep(freq, 300)
                        time.sleep(0.1)
                    
                    # Send alert
                    self.alert_sender.send_alert("voice", message, keywords=keywords)
                    self.alert_sender.update_status(state="VOICE_DETECTED")
                    
        except Exception as e:
            pass

    def run(self):
        """Main voice detection loop"""
        print("\n   🎤 Listening for emergency keywords...")
        print("      English: help, emergency, fall, hurt")
        print("      Malayalam: സഹായം, വീഴ്ച, വേദന")
        print("      Hindi: मदद, गिर गया, चोट")
        print("      " + "-" * 40)
        
        self.alert_sender.update_status(voice_active=True)
        
        with self.microphone as source:
            while self.running:
                try:
                    # Listen for audio
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    
                    # Queue for processing
                    if audio:
                        self.audio_queue.put(audio)
                        
                except sr.WaitTimeoutError:
                    continue
                except Exception as e:
                    time.sleep(0.5)
        
        self.alert_sender.update_status(voice_active=False)

    def stop(self):
        self.running = False

# ==================== MAIN ====================
def main():
    print("\n" + "="*70)
    print("🚀 GUARDIAN NET - INTEGRATED EMERGENCY DETECTOR")
    print("="*70)
    
    # Get patient ID from user or use default
    try:
        patient_id = input("Enter patient ID (default: 1): ").strip()
        patient_id = int(patient_id) if patient_id else 1
    except:
        patient_id = 1
    
    print(f"\n📱 Patient ID: {patient_id}")
    print("="*70)
    
    # Initialize alert sender
    alert_sender = AlertSender(patient_id=patient_id)
    
    # Test server connection
    try:
        requests.get(SERVER_URL, timeout=2)
        print("✅ Connected to Guardian Net server")
    except:
        print("⚠️ Cannot connect to server - running in local mode")
    
    print("\n🔧 Initializing detectors...")
    
    # Initialize detectors
    fall_detector = FallDetector(alert_sender)
    voice_detector = VoiceDetector(alert_sender)
    
    print("\n" + "="*70)
    print("✅ ALL DETECTORS READY - Starting...")
    print("="*70)
    print("📹 Fall Detection: Camera window open")
    print("🎤 Voice Detection: Listening in background")
    print("\nPress 'q' in video window to quit")
    print("Press 'r' in video window to reset counter")
    print("="*70 + "\n")
    
    # Start voice detection in background
    voice_thread = threading.Thread(target=voice_detector.run, daemon=True)
    voice_thread.start()
    
    try:
        # Run fall detection in main thread
        fall_detector.run()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")
    finally:
        fall_detector.stop()
        voice_detector.stop()
        alert_sender.stop()
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Falls detected: {fall_detector.total_falls}")
    print(f"   Voice emergencies: {voice_detector.emergency_count}")
    print(f"   Alerts sent: {alert_sender.alert_count}")
    print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()