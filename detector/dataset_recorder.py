#!/usr/bin/env python
"""
Dataset Recorder for Human Fall Detection
════════════════════════════════════════════════════════════════
Records real-time detection data and exports to CSV/JSON format
for training and analysis purposes.
════════════════════════════════════════════════════════════════
"""

import os
import csv
import json
import time
import uuid
import numpy as np
from datetime import datetime
from collections import deque
from threading import Lock


class DatasetRecorder:
    """
    Records fall detection data for dataset creation.
    
    Captures per-frame features that are useful for:
    - Training ML models
    - Analyzing fall patterns
    - Benchmarking detection algorithms
    """
    
    def __init__(self, output_dir="fall_dataset", session_id=None):
        self.output_dir = output_dir
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.recording_id = str(uuid.uuid4())[:8]
        
        # Create output directories
        self.frames_dir = os.path.join(output_dir, "frames", self.session_id)
        self.annotations_dir = os.path.join(output_dir, "annotations", self.session_id)
        
        os.makedirs(self.frames_dir, exist_ok=True)
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Data storage
        self.frame_data = []
        self.event_data = []  # Fall events, voice alerts, etc.
        self.metadata = {
            'session_id': self.session_id,
            'recording_id': self.recording_id,
            'start_time': time.time(),
            'total_frames': 0,
            'total_falls': 0,
            'total_voice_alerts': 0,
        }
        
        # Thread safety
        self.lock = Lock()
        
        # Sliding windows for temporal features
        self.velocity_window = deque(maxlen=5)
        self.angle_window = deque(maxlen=5)
        self.confidence_window = deque(maxlen=5)
        
        # Previous frame data
        self.prev_bbox = None
        self.prev_time = None
        self.prev_height = None
        
        # Recording state
        self.is_recording = True
        self.paused = False
        
        # CSV file handle
        self.csv_file = None
        self.csv_writer = None
        self._init_csv()
        
        print(f"   📁 Dataset recorder initialized")
        print(f"   📂 Output: {self.output_dir}/{self.session_id}")
    
    def _init_csv(self):
        """Initialize CSV file with headers for frame data."""
        csv_path = os.path.join(self.annotations_dir, "frame_data.csv")
        
        self.csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Define CSV headers - these are your FEATURE COLUMNS for ML training
        headers = [
            # Frame identification
            'frame_id',
            'timestamp',
            'elapsed_time_sec',
            
            # Bounding box (normalized 0-1)
            'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2',
            'bbox_width', 'bbox_height',
            'bbox_center_x', 'bbox_center_y',
            
            # Aspect ratio features
            'aspect_ratio',
            'aspect_ratio_category',  # standing=0, sitting=1, fallen=2
            
            # Body angle features
            'body_angle_deg',
            'angle_category',  # standing=0, sitting=1, fallen=2
            
            # Motion features
            'velocity_x', 'velocity_y', 'velocity_magnitude',
            'height_loss_ratio',  # How much height was lost
            'height_loss_category',  # sudden=1, gradual=0
            
            # Confidence scores
            'fall_confidence',
            'stand_confidence',
            'avg_fall_confidence_5frames',  # Smoothed confidence
            'avg_angle_5frames',
            
            # Position features (normalized)
            'center_y_normalized',  # Lower = closer to ground
            'distance_to_ground',
            
            # Detection state
            'is_fall_detected',
            'is_sitting',
            'is_standing',
            
            # Additional features for ML
            'frame_quality_score',
            'person_in_frame',
            
            # Ground truth label (can be manually annotated later)
            'label',  # 0=standing, 1=sitting, 2=fallen, -1=unknown
            'label_source',  # 'auto', 'manual', 'model'
        ]
        
        self.csv_writer.writerow(headers)
        self.csv_file.flush()
    
    def _calculate_features(self, bbox, fall_conf, stand_conf, body_angle, state):
        """Calculate all features for a single frame."""
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Aspect ratio
        ar = h / w
        
        # Angle category
        if body_angle < 42:
            angle_cat = 0  # standing
        elif body_angle < 44:
            angle_cat = 1  # sitting
        else:
            angle_cat = 2  # fallen
        
        # Aspect ratio category
        if ar > 1.80:
            ar_cat = 0  # standing
        elif ar > 0.90:
            ar_cat = 1  # sitting
        else:
            ar_cat = 2  # fallen
        
        # Motion features
        now = time.time()
        vx = vy = vel = 0.0
        h_loss = 0.0
        
        if self.prev_bbox is not None and self.prev_time is not None:
            dt = max(0.02, now - self.prev_time)
            vx = (cx - (self.prev_bbox[0] + self.prev_bbox[2]) / 2) / dt
            vy = (cy - (self.prev_bbox[1] + self.prev_bbox[3]) / 2) / dt
            vel = np.sqrt(vx**2 + vy**2)
            
            if self.prev_height is not None:
                h_loss = (self.prev_height - h) / max(1, self.prev_height)
        
        # Height loss category
        h_loss_cat = 1 if h_loss > 0.18 else 0
        
        # Normalized position (lower y = closer to ground)
        center_y_norm = cy / 480  # Assuming FRAME_H = 480
        dist_to_ground = 1.0 - center_y_norm
        
        # State flags
        is_fall = 1 if state == "FALL_DETECTED" else 0
        is_sitting = 1 if (ar_cat == 1 or angle_cat == 1) else 0
        is_standing = 1 if (ar_cat == 0 and angle_cat == 0) else 0
        
        # Frame quality (placeholder - can add blur detection)
        frame_quality = 1.0
        
        # Smoothed features
        self.confidence_window.append(fall_conf)
        self.angle_window.append(body_angle)
        
        avg_conf_5 = np.mean(self.confidence_window) if len(self.confidence_window) >= 2 else fall_conf
        avg_angle_5 = np.mean(self.angle_window) if len(self.angle_window) >= 2 else body_angle
        
        return {
            'bbox_x1': x1 / 640,
            'bbox_y1': y1 / 480,
            'bbox_x2': x2 / 640,
            'bbox_y2': y2 / 480,
            'bbox_width': w / 640,
            'bbox_height': h / 480,
            'bbox_center_x': cx / 640,
            'bbox_center_y': cy / 480,
            'aspect_ratio': ar,
            'aspect_ratio_category': ar_cat,
            'body_angle_deg': body_angle,
            'angle_category': angle_cat,
            'velocity_x': vx,
            'velocity_y': vy,
            'velocity_magnitude': vel,
            'height_loss_ratio': h_loss,
            'height_loss_category': h_loss_cat,
            'fall_confidence': fall_conf,
            'stand_confidence': stand_conf,
            'avg_fall_confidence_5frames': avg_conf_5,
            'avg_angle_5frames': avg_angle_5,
            'center_y_normalized': center_y_norm,
            'distance_to_ground': dist_to_ground,
            'is_fall_detected': is_fall,
            'is_sitting': is_sitting,
            'is_standing': is_standing,
            'frame_quality_score': frame_quality,
            'person_in_frame': 1,
        }
    
    def record_frame(self, bbox, fall_conf, stand_conf, body_angle, state, frame_id=None):
        """Record detection data for a single frame."""
        if not self.is_recording or self.paused:
            return
        
        frame_id = frame_id or len(self.frame_data) + 1
        timestamp = time.time()
        elapsed = timestamp - self.metadata['start_time']
        
        features = self._calculate_features(bbox, fall_conf, stand_conf, body_angle, state)
        
        if features is None:
            # No person detected - record empty frame
            row = [frame_id, timestamp, elapsed] + [0.0] * 28 + [-1, 'auto']
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            return
        
        # Determine label based on detection
        if features['is_fall_detected']:
            label = 2  # fallen
            label_source = 'auto'
        elif features['is_sitting']:
            label = 1  # sitting
            label_source = 'auto'
        elif features['is_standing']:
            label = 0  # standing
            label_source = 'auto'
        else:
            label = -1  # unknown
            label_source = 'auto'
        
        # Build CSV row
        row = [
            frame_id,
            timestamp,
            elapsed,
            features['bbox_x1'],
            features['bbox_y1'],
            features['bbox_x2'],
            features['bbox_y2'],
            features['bbox_width'],
            features['bbox_height'],
            features['bbox_center_x'],
            features['bbox_center_y'],
            features['aspect_ratio'],
            features['aspect_ratio_category'],
            features['body_angle_deg'],
            features['angle_category'],
            features['velocity_x'],
            features['velocity_y'],
            features['velocity_magnitude'],
            features['height_loss_ratio'],
            features['height_loss_category'],
            features['fall_confidence'],
            features['stand_confidence'],
            features['avg_fall_confidence_5frames'],
            features['avg_angle_5frames'],
            features['center_y_normalized'],
            features['distance_to_ground'],
            features['is_fall_detected'],
            features['is_sitting'],
            features['is_standing'],
            features['frame_quality_score'],
            features['person_in_frame'],
            label,
            label_source,
        ]
        
        with self.lock:
            self.csv_writer.writerow(row)
            self.csv_file.flush()
            
            self.frame_data.append({
                'frame_id': frame_id,
                'timestamp': timestamp,
                'elapsed': elapsed,
                'bbox': bbox,
                'fall_confidence': fall_conf,
                'stand_confidence': stand_conf,
                'body_angle': body_angle,
                'state': state,
                'label': label,
                'features': features,
            })
            
            self.metadata['total_frames'] += 1
        
        # Update previous values for motion calculation
        if bbox is not None:
            self.prev_bbox = bbox
            self.prev_time = timestamp
            self.prev_height = max(1, bbox[3] - bbox[1])
    
    def record_fall_event(self, fall_conf, body_angle, total_falls, bbox=None):
        """Record when a fall is detected - key event for dataset."""
        event = {
            'event_type': 'fall_detected',
            'timestamp': time.time(),
            'confidence': fall_conf,
            'body_angle': body_angle,
            'event_number': total_falls,
            'bbox': bbox,
        }
        
        with self.lock:
            self.event_data.append(event)
            self.metadata['total_falls'] += 1
        
        print(f"   📝 [DATASET] Fall event #{total_falls} recorded")
    
    def record_voice_event(self, text, keywords, total_alerts):
        """Record voice emergency event."""
        event = {
            'event_type': 'voice_emergency',
            'timestamp': time.time(),
            'transcribed_text': text,
            'keywords_detected': keywords,
            'event_number': total_alerts,
        }
        
        with self.lock:
            self.event_data.append(event)
            self.metadata['total_voice_alerts'] += 1
        
        print(f"   📝 [DATASET] Voice event recorded: {keywords}")
    
    def export_to_json(self, filepath=None):
        """Export all recorded data to JSON format."""
        if filepath is None:
            filepath = os.path.join(self.annotations_dir, "full_dataset.json")
        
        data = {
            'metadata': self.metadata,
            'end_time': time.time(),
            'total_frames': len(self.frame_data),
            'total_events': len(self.event_data),
            'frame_data': self.frame_data,
            'event_data': self.event_data,
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"   💾 JSON export: {filepath}")
        return filepath
    
    def export_for_ml_training(self, output_dir=None):
        """
        Export data in format ready for ML training.
        Creates separate train/test splits.
        """
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "ml_ready", self.session_id)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Separate by label
        standing = [f for f in self.frame_data if f['label'] == 0]
        sitting = [f for f in self.frame_data if f['label'] == 1]
        fallen = [f for f in self.frame_data if f['label'] == 2]
        
        print(f"   📊 Data distribution:")
        print(f"      Standing: {len(standing)} frames")
        print(f"      Sitting:  {len(sitting)} frames")
        print(f"      Fallen:   {len(fallen)} frames")
        
        # Export each class separately
        for name, data in [('standing', standing), ('sitting', sitting), ('fallen', fallen)]:
            if data:
                csv_path = os.path.join(output_dir, f"{name}.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # Headers for ML features
                    headers = [
                        'bbox_width', 'bbox_height', 'aspect_ratio',
                        'body_angle_deg', 'velocity_magnitude', 'height_loss_ratio',
                        'fall_confidence', 'center_y_normalized', 'label'
                    ]
                    writer.writerow(headers)
                    
                    for frame in data:
                        feat = frame['features']
                        writer.writerow([
                            feat['bbox_width'],
                            feat['bbox_height'],
                            feat['aspect_ratio'],
                            feat['body_angle_deg'],
                            feat['velocity_magnitude'],
                            feat['height_loss_ratio'],
                            feat['fall_confidence'],
                            feat['center_y_normalized'],
                            frame['label'],
                        ])
                
                print(f"      ✓ {name}.csv ({len(data)} samples)")
        
        return output_dir
    
    def close(self):
        """Close recorder and finalize files."""
        if self.csv_file:
            self.csv_file.close()
        
        # Export summary
        summary_path = os.path.join(self.annotations_dir, "session_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metadata': self.metadata,
                'total_frames_recorded': len(self.frame_data),
                'total_events': len(self.event_data),
                'session_duration_sec': time.time() - self.metadata['start_time'],
            }, f, indent=2)
        
        print(f"\n   📁 Dataset saved to: {self.annotations_dir}")
        print(f"   📊 Frames recorded: {len(self.frame_data)}")
        print(f"   📊 Events recorded: {len(self.event_data)}")
    
    def pause(self):
        """Pause recording."""
        self.paused = True
        print("   ⏸️  Dataset recording paused")
    
    def resume(self):
        """Resume recording."""
        self.paused = False
        print("   ▶️  Dataset recording resumed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION HELPER - How to add to guardian_all.py
# ══════════════════════════════════════════════════════════════════════════════
"""
To integrate DatasetRecorder into guardian_all.py:

1. Add recorder to main():
   
   from dataset_recorder import DatasetRecorder
   
   # After alert_sender initialization
   recorder = DatasetRecorder(output_dir="fall_dataset", session_id=f"patient_{patient_id}")
   
   # Pass recorder to detectors
   fall_detector = UnifiedFallDetector(alert_sender, shared_state, recorder)

2. Modify UnifiedFallDetector to use recorder:
   
   class UnifiedFallDetector:
       def __init__(self, alert_sender, shared_state, recorder=None):
           ...
           self.recorder = recorder
       
       def process_frame(self, frame):
           ...
           # After getting detection results
           if self.recorder:
               self.recorder.record_frame(
                   bbox=bbox,
                   fall_conf=fall_conf,
                   stand_conf=stand_conf,
                   body_angle=self.body_angle,
                   state=self.state,
               )
           ...

3. Record fall events:
   
   if self.state == "FALL_DETECTED" and self.consecutive_fall_frms == REQUIRED_FALL_FRM:
       if self.recorder:
           self.recorder.record_fall_event(
               fall_conf=fall_conf,
               body_angle=self.body_angle,
               total_falls=self.total_falls,
               bbox=bbox,
           )

4. At program exit:
   
   recorder.close()
   recorder.export_to_json()
   recorder.export_for_ml_training()

═════════════════════════════════════════════════════════════════════════════
"""
