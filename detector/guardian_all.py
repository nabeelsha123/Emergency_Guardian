#!/usr/bin/env python
"""
Guardian Net - Enhanced Fall Detection + Voice Detection (Fixed)
════════════════════════════════════════════════════════════════
UPDATED: Clean terminal output - only shows fall alerts
         + Real-time dataset collection (auto-saves training images)
         + Fall video recording (saves video + snapshot on every fall)
         + RETRAIN HELPER - suggests retraining after 50 new samples

CHANGES v2:
  • Fall video clips capped at 10 seconds (150 frames @ 15 fps)
  • Dataset collector saves ONLY fallen frames (sitting/standing removed)
  • RetrainHelper watches dataset and prints retrain command
════════════════════════════════════════════════════════════════
"""

# ── Suppress OpenCV noise ──────────────────────────────────────────────────────
import os
os.environ['OPENCV_LOG_LEVEL']     = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
try:
    cv2.setLogLevel(0)
except AttributeError:
    pass

import numpy as np
import time
import threading
import queue
import sys
import warnings
import math
import platform
import json
import shutil
from datetime import datetime
from collections import deque
warnings.filterwarnings('ignore')

# ── Guardian integration ───────────────────────────────────────────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from guardian_integration import GuardianAlertSender
    GUARDIAN_AVAILABLE = True
except ImportError:
    GUARDIAN_AVAILABLE = False

    class GuardianAlertSender:
        def __init__(self, patient_id=1):
            self.patient_id  = patient_id
            self.alert_count = 0
            print("⚠️  guardian_integration not found — alerts logged only.")
        def test_connection(self):
            return False
        def send_alert(self, alert_type, message, confidence=None):
            self.alert_count += 1
            ts     = time.strftime("%H:%M:%S")
            suffix = f" | conf={confidence:.2%}" if confidence is not None else ""
            print(f"[ALERT {ts}] {alert_type.upper()}{suffix} | {message}")

# ── YOLO ───────────────────────────────────────────────────────────────────────
from ultralytics import YOLO

# ── Voice ──────────────────────────────────────────────────────────────────────
import speech_recognition as sr
import re

# ── Cross-platform alarm  (non-blocking) ──────────────────────────────────────
def play_alarm_sound(kind="fall"):
    """Play alarm in a daemon thread so it never blocks detection loops."""
    patterns = {
        "fall":  [(1200,180),(900,180),(1200,180),(900,180),(1500,500)],
        "voice": [(1000,200),(1200,200),(1000,400)],
    }
    chosen = patterns.get(kind, patterns["fall"])

    def _play():
        if platform.system() == "Windows":
            import winsound
            for freq, dur in chosen:
                winsound.Beep(freq, dur)
                time.sleep(0.04)
        elif platform.system() == "Darwin":
            os.system(f'say "Emergency {kind} alert"')
        else:
            os.system('paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga'
                      ' 2>/dev/null || aplay /usr/share/sounds/alsa/Front_Left.wav'
                      ' 2>/dev/null || true')

    threading.Thread(target=_play, daemon=True).start()

# ── Shared alert queue ─────────────────────────────────────────────────────────
alert_queue = queue.Queue()

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS — IMPROVED ANGLE DETECTION with LOWERED CONFIDENCE
# ══════════════════════════════════════════════════════════════════════════════
FRAME_W, FRAME_H   = 640, 480

# LOWERED CONFIDENCE THRESHOLD for better fall detection
FALL_CONF_THRESH   = 0.69
REQUIRED_FALL_FRM  = 5
REQUIRED_STAND_FRM = 10
ALERT_COOLDOWN     = 3

# IMPROVED ANGLE-BASED DETECTION
ANGLE_FALL_MIN     = 48
ANGLE_FALL_HIGH    = 69
ANGLE_STAND_MAX    = 46
ANGLE_SITTING_MAX  = 55

# Aspect ratio thresholds
AR_FALL_MAX        = 0.90
AR_FALL_ZONE_MAX   = 1.20
AR_STAND_MIN       = 1.80

# Motion detection thresholds
VELOCITY_FALL_MIN  = 0.16
HEIGHT_LOSS_FALL   = 0.18

CLASS_NAMES        = ["fallen", "sitting", "standing"]

# ── CHANGED: Maximum frames to record per fall clip (10 s × 15 fps = 150) ──
FALL_RECORD_MAX_FRAMES = 150   # ← NEW constant (10-second cap)


# ══════════════════════════════════════════════════════════════════════════════
#  FALL RECORDER  ← records video + snapshot while FALL_DETECTED is active
#
#  Saves into:  realtime_dataset/fall_recordings/
#    fall_YYYYMMDD_HHMMSS.avi   ← video of the fall event (max 10 s)
#    fall_YYYYMMDD_HHMMSS.jpg   ← snapshot of the first detected frame
#
#  CHANGE: recording auto-stops after FALL_RECORD_MAX_FRAMES (150 frames)
#          so clips are always ≤ 10 seconds regardless of how long the fall
#          state persists.
# ══════════════════════════════════════════════════════════════════════════════
class FallRecorder:
    """
    Records a video clip (max 10 s) and saves a snapshot for every fall event.
    Runs its own background write-thread so it never stalls detection.
    """

    def __init__(self, base_dir):
        # Save inside realtime_dataset/fall_recordings/
        self.save_dir = os.path.join(base_dir, "fall_recordings")
        os.makedirs(self.save_dir, exist_ok=True)

        self._writer      = None     # cv2.VideoWriter
        self._lock        = threading.Lock()
        self._recording   = False
        self._current_stem = None
        self._frame_count = 0
        self._total_clips = 0

        # Count existing clips on startup
        existing = [f for f in os.listdir(self.save_dir)
                    if f.endswith(".avi")]
        self._total_clips = len(existing)

        print(f"   🎥 Fall recorder ready → {self.save_dir}")
        print(f"      Existing clips: {self._total_clips}")
        print(f"      Max clip length: {FALL_RECORD_MAX_FRAMES/15:.0f}s "
              f"({FALL_RECORD_MAX_FRAMES} frames @ 15 fps)")

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self, frame):
        """
        Called exactly ONCE when FALL_DETECTED state is entered.
        Creates a new VideoWriter + saves the first frame as snapshot.
        """
        with self._lock:
            if self._recording:
                return   # already recording (safety guard)

            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = f"fall_{ts}"
            self._current_stem = stem

            # ── Video writer ──────────────────────────────────
            vid_path = os.path.join(self.save_dir, f"{stem}.avi")
            fourcc   = cv2.VideoWriter_fourcc(*"XVID")
            self._writer = cv2.VideoWriter(
                vid_path, fourcc, 15.0, (FRAME_W, FRAME_H)
            )

            # ── Snapshot of first fall frame ──────────────────
            snap_path = os.path.join(self.save_dir, f"{stem}.jpg")
            snap = cv2.resize(frame, (FRAME_W, FRAME_H))
            cv2.imwrite(snap_path, snap,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

            self._recording   = True
            self._frame_count = 1   # snapshot counts as frame 1

            print(f"\n   🎥 Fall recording started → {stem}.avi  "
                  f"(max {FALL_RECORD_MAX_FRAMES/15:.0f}s)")
            print(f"   📸 Snapshot saved       → {stem}.jpg")

    def write(self, frame):
        """
        Called every frame while FALL_DETECTED is active.
        Thread-safe; silently skipped if not currently recording.

        CHANGE: auto-stops after FALL_RECORD_MAX_FRAMES so the clip
                is always capped at 10 seconds.
        """
        with self._lock:
            if not self._recording or self._writer is None:
                return

            # ── CHANGED: cap at FALL_RECORD_MAX_FRAMES ─────────────────────
            if self._frame_count >= FALL_RECORD_MAX_FRAMES:
                # Silently finalise — fall state may still be active but we
                # stop recording to keep clip ≤ 10 s.
                self._finalise()
                return

            resized = cv2.resize(frame, (FRAME_W, FRAME_H))
            self._writer.write(resized)
            self._frame_count += 1

    def stop(self):
        """
        Called when fall state clears (person got up or timeout).
        Finalises and closes the video file (no-op if already auto-stopped).
        """
        with self._lock:
            self._finalise()

    def _finalise(self):
        """
        Internal: close the VideoWriter and update counters.
        Must be called with self._lock already held.
        """
        if not self._recording:
            return

        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._total_clips += 1
        duration = self._frame_count / 15.0   # approx seconds at 15 fps
        print(f"   🎥 Fall recording saved  "
              f"({self._frame_count} frames, ~{duration:.1f}s) | "
              f"Total clips: {self._total_clips}")

        self._recording    = False
        self._current_stem = None
        self._frame_count  = 0

    # ── Info helpers ───────────────────────────────────────────────────────

    @property
    def is_recording(self):
        return self._recording

    def print_summary(self):
        print(f"\n   🎥 Fall Recordings → {self.save_dir}")
        print(f"      Total clips saved: {self._total_clips}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATASET COLLECTOR — saves FALL frames + YOLO labels automatically
#
#  CHANGE: sitting and standing background collection REMOVED entirely.
#          Only fallen (class 0) frames are saved to keep the dataset
#          focused and disk usage minimal.
# ══════════════════════════════════════════════════════════════════════════════
class DatasetCollector:
    """
    Automatically saves FALL frames to a YOLO-format dataset.

    Directory layout created:
        realtime_dataset/
            images/
                train/   ← 80 % of captures go here
                val/     ← 20 % of captures go here
            labels/
                train/
                val/
            dataset_log.jsonl   ← one JSON record per saved sample
            dataset.yaml        ← ready-to-use YOLO training config

    Class map  →  0=fallen  (sitting and standing no longer collected)
    """

    CLASS_MAP = {0: "fallen", 1: "sitting", 2: "standing"}

    # How many frames to skip between saves (avoid duplicate fall frames)
    SAVE_EVERY_N_FRAMES = 8

    # Minimum fall-confidence required to save a "fallen" sample
    MIN_FALL_CONF_TO_SAVE = 0.55

    def __init__(self, base_dir=None):
        if base_dir is None:
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "realtime_dataset"
            )
        self.base_dir = base_dir

        # Build directory tree
        for split in ("train", "val"):
            os.makedirs(os.path.join(base_dir, "images", split), exist_ok=True)
            os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

        self.log_path  = os.path.join(base_dir, "dataset_log.jsonl")
        self.yaml_path = os.path.join(base_dir, "dataset.yaml")
        self._write_yaml()

        # Internal counters — only track fallen (0)
        self._total_saved    = {0: 0, 1: 0, 2: 0}
        self._frame_counter  = 0
        self._lock           = threading.Lock()

        # Load existing counts from log if dataset already exists
        self._load_existing_counts()

        print(f"   💾 Dataset collector ready → {base_dir}")
        print(f"      Existing fall samples: {self._total_saved[0]}")
        print(f"      ℹ️  Only FALL frames are collected (sitting/standing disabled)")

    # ── Internal helpers ───────────────────────────────────────────────────

    def _write_yaml(self):
        """Write / refresh dataset.yaml for YOLO training."""
        yaml_content = (
            f"# Auto-generated by Guardian Net real-time dataset collector\n"
            f"path: {self.base_dir}\n"
            f"train: images/train\n"
            f"val:   images/val\n"
            f"\nnc: 3\n"
            f"names: ['fallen', 'sitting', 'standing']\n"
        )
        with open(self.yaml_path, "w") as f:
            f.write(yaml_content)

    def _load_existing_counts(self):
        """Read existing log to resume counters after restart."""
        if not os.path.exists(self.log_path):
            return
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        cls = rec.get("class_id", -1)
                        if cls in self._total_saved:
                            self._total_saved[cls] += 1
                    except Exception:
                        pass
        except Exception:
            pass

    def _choose_split(self):
        """80 / 20 train / val split."""
        total = sum(self._total_saved.values())
        return "val" if total % 5 == 4 else "train"

    def _make_filename(self, class_id, split):
        ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label = self.CLASS_MAP[class_id]
        stem  = f"{label}_{ts}"
        img   = os.path.join(self.base_dir, "images", split, f"{stem}.jpg")
        lbl   = os.path.join(self.base_dir, "labels", split, f"{stem}.txt")
        return img, lbl, stem

    def _save_yolo_label(self, lbl_path, class_id, bbox, frame_w, frame_h):
        """
        Write a single YOLO annotation line.
        bbox = (x1, y1, x2, y2) in pixel coords.
        If bbox is None a full-frame box is written as fallback.
        """
        if bbox is not None:
            x1, y1, x2, y2 = bbox
        else:
            x1, y1, x2, y2 = 0, 0, frame_w, frame_h

        cx = ((x1 + x2) / 2) / frame_w
        cy = ((y1 + y2) / 2) / frame_h
        bw = (x2 - x1) / frame_w
        bh = (y2 - y1) / frame_h

        # Clamp to [0, 1]
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))

        with open(lbl_path, "w") as f:
            f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _log_record(self, stem, split, class_id, confidence, angle):
        record = {
            "file":       stem,
            "split":      split,
            "class_id":   class_id,
            "class_name": self.CLASS_MAP[class_id],
            "confidence": round(confidence, 4),
            "angle":      angle,
            "timestamp":  datetime.now().isoformat(),
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ── Public API ─────────────────────────────────────────────────────────

    def try_save(self, frame, detector_state, fall_conf, stand_conf,
                 bbox, cls_id, cls_conf, body_angle):
        """
        Called every frame from process_frame().
        CHANGE: only saves FALL frames — sitting and standing paths removed.

        Parameters
        ----------
        frame          : raw BGR frame (will be resized to FRAME_W x FRAME_H)
        detector_state : "MONITORING" | "FALL_DETECTED"
        fall_conf      : float 0-1
        stand_conf     : float 0-1  (unused now, kept for signature compat)
        bbox           : (x1,y1,x2,y2) or None
        cls_id         : int YOLO class id or None
        cls_conf       : float YOLO class confidence
        body_angle     : estimated body angle in degrees
        """
        with self._lock:
            self._frame_counter += 1

            # ── Save FALL frames only ──────────────────────────────────────
            if (detector_state == "FALL_DETECTED"
                    and fall_conf >= self.MIN_FALL_CONF_TO_SAVE
                    and self._frame_counter % self.SAVE_EVERY_N_FRAMES == 0):
                self._do_save(frame, class_id=0, bbox=bbox,
                              confidence=fall_conf, angle=body_angle)

            # ── Sitting / standing collection intentionally removed ────────

    def _do_save(self, frame, class_id, bbox, confidence, angle):
        """Internal: resize, write image + label, update counters."""
        try:
            split = self._choose_split()
            img_path, lbl_path, stem = self._make_filename(class_id, split)

            # Resize frame to model input size
            save_frame = cv2.resize(frame, (FRAME_W, FRAME_H))
            cv2.imwrite(img_path, save_frame, [cv2.IMWRITE_JPEG_QUALITY, 92])

            self._save_yolo_label(lbl_path, class_id, bbox, FRAME_W, FRAME_H)
            self._log_record(stem, split, class_id, confidence, angle)

            self._total_saved[class_id] += 1
            label = self.CLASS_MAP[class_id]
            total = self._total_saved[0]   # only fallen now
            print(f"   📸 Dataset +1 [{label.upper()}] conf={confidence:.2%} "
                  f"angle={int(angle)}° | fallen_total={total}")

        except Exception as e:
            print(f"   ⚠️  Dataset save error: {e}")

    def get_summary(self):
        return dict(self._total_saved)

    def print_summary(self):
        s = self._total_saved
        print(f"\n   📊 Dataset Summary → {self.base_dir}")
        print(f"      fallen={s[0]}  (sitting/standing not collected)")
        print(f"      Config: {self.yaml_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  RETRAIN HELPER — tells operator when to retrain with new data
# ══════════════════════════════════════════════════════════════════════════════
class RetrainHelper(threading.Thread):
    """
    Background thread that watches the dataset fall count and prints
    a retrain command every time NOTIFY_EVERY new samples are saved.

    HOW REAL-TIME DATA BECOMES THE NEXT MODEL:
      1.  This system saves fall frames → realtime_dataset/
      2.  You retrain:  yolo train data=realtime_dataset/dataset.yaml
                        model=<current_best.pt>  pretrained=True
      3.  New best.pt auto-loads on next startup via _load_model()
    """
    NOTIFY_EVERY = 50   # print command after every 50 new fall frames

    def __init__(self, dataset_collector, model_path):
        super().__init__(daemon=True, name="RetrainHelper")
        self.dataset = dataset_collector
        self.model_path = model_path
        self._last_notified = 0

    def run(self):
        while True:
            time.sleep(30)
            fallen = self.dataset._total_saved[0]
            if fallen >= self._last_notified + self.NOTIFY_EVERY:
                self._last_notified = (fallen // self.NOTIFY_EVERY) * self.NOTIFY_EVERY
                yaml = self.dataset.yaml_path
                print(f"\n{'='*60}")
                print(f"💡 RETRAIN SUGGESTION — {fallen} fall samples collected")
                print(f"   Run this command to fine-tune your model:")
                print(f"   yolo train data={yaml} \\")
                print(f"        model={self.model_path} \\")
                print(f"        pretrained=True epochs=50 imgsz=640")
                print(f"   Then restart Guardian Net — it will auto-load the new model.")
                print(f"{'='*60}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  FALL DETECTOR — CLEAN OUTPUT (no angle logging) + DATASET INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
class UnifiedFallDetector:
    def __init__(self, alert_sender, shared_state):
        print("   📹 Initializing Enhanced Fall Detection...")
        self.alert_sender = alert_sender
        self.shared_state = shared_state
        self.model, self.use_custom, self.model_label = self._load_model()

        self.state                  = "MONITORING"
        self.total_falls            = 0
        self.consecutive_fall_frms  = 0
        self.consecutive_stand_frms = 0
        self.fall_start_time        = 0.0
        self.last_alert_time        = 0.0

        self.fall_hist     = deque(maxlen=8)
        self.stand_hist    = deque(maxlen=8)
        self.angle_hist    = deque(maxlen=6)
        self.velocity_hist = deque(maxlen=8)
        self.height_hist   = deque(maxlen=20)
        self.body_angle    = 0.0
        self.prev_bbox     = None
        self.prev_time     = None
        self.running       = True
        self.frame_queue   = queue.Queue(maxsize=2)

        # ── Dataset collector (saves fall frames for future training) ───────
        self.dataset = DatasetCollector()

        # ── Fall recorder (saves video clips of falls) ──────────────────────
        self.recorder = FallRecorder(base_dir=self.dataset.base_dir)

        # ── Retrain helper (suggests when to retrain) ───────────────────────
        self.retrain_helper = RetrainHelper(self.dataset, self._get_model_path())
        self.retrain_helper.start()

        print(f"   ✅ Fall Detection Ready [{self.model_label}]")
        print(f"   ⚙️  Fall threshold: {FALL_CONF_THRESH} | Angle threshold: {ANGLE_FALL_MIN}°")

    def _get_model_path(self):
        """Get current model path for retraining"""
        base = os.path.dirname(os.path.abspath(__file__))
        paths = [
            os.path.join(base, "best.pt"),
            os.path.join(os.path.dirname(base), "runs", "train", "fall_detection", "weights", "best.pt"),
        ]
        for p in paths:
            if os.path.exists(p):
                return p
        return "yolov8n.pt"

    def _load_model(self):
        base  = os.path.dirname(os.path.abspath(__file__))
        root  = os.path.dirname(base)
        paths = [
            os.path.join(root,"runs","train","fall_custom_scratch","weights","best.pt"),
            os.path.join(root,"runs","train","fall_detection",     "weights","best.pt"),
            os.path.join(base,"best.pt"),
        ]
        for p in paths:
            if os.path.exists(p):
                print(f"   ✅ Custom model → {p}")
                return YOLO(p), True, "CUSTOM"
        if not os.path.exists(os.path.join(base,"yolov8n-pose.pt")):
            print("   📥 Downloading yolov8n-pose.pt …")
        return YOLO("yolov8n-pose.pt"), False, "POSE"

    def _calculate_body_angle_from_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect_ratio = height / width
        
        if aspect_ratio > 2.0:
            angle = 10.0
        elif aspect_ratio > 1.5:
            angle = 20.0 - (aspect_ratio - 1.5) * 20.0
        elif aspect_ratio > 0.9:
            angle = 60.0 - (aspect_ratio - 0.9) * (40.0 / 0.6)
        else:
            angle = 75.0 + (1.0 - aspect_ratio) * 25.0
            angle = min(90.0, angle)
        
        self.body_angle = angle
        self.angle_hist.append(angle)
        smoothed_angle = np.mean(self.angle_hist) if len(self.angle_hist) >= 2 else angle
        
        return smoothed_angle

    def _get_fall_confidence_from_angle(self, angle):
        if angle <= ANGLE_STAND_MAX:
            return 0.0
        elif angle <= ANGLE_SITTING_MAX:
            return 0.1
        elif angle < ANGLE_FALL_MIN:
            return (angle - ANGLE_SITTING_MAX) / (ANGLE_FALL_MIN - ANGLE_SITTING_MAX) * 0.4
        elif angle <= ANGLE_FALL_HIGH:
            return 0.6 + (angle - ANGLE_FALL_MIN) / (ANGLE_FALL_HIGH - ANGLE_FALL_MIN) * 0.3
        else:
            return 0.9 + min(0.1, (angle - ANGLE_FALL_HIGH) / 30.0)

    def _fall_conf_bbox(self, bbox, cls_id, cls_conf):
        x1, y1, x2, y2 = bbox
        w_b = max(1, x2 - x1)
        h_b = max(1, y2 - y1)
        scores = []

        angle = self._calculate_body_angle_from_bbox(bbox)
        angle_score = self._get_fall_confidence_from_angle(angle)
        scores.append(angle_score * 0.40)

        ar = h_b / w_b
        if ar < AR_FALL_MAX:
            ar_score = 1.0
        elif ar < AR_FALL_ZONE_MAX:
            ar_score = 1.0 - (ar - AR_FALL_MAX) / (AR_FALL_ZONE_MAX - AR_FALL_MAX)
        else:
            ar_score = 0.0
        scores.append(ar_score * 0.20)

        cy_norm = ((y1 + y2) / 2) / FRAME_H
        if cy_norm > 0.55:
            ground_score = min(1.0, (cy_norm - 0.45) * 2.0)
            scores.append(ground_score * 0.15)

        if self.prev_bbox is not None and self.prev_time is not None:
            dt = max(0.02, time.time() - self.prev_time)
            ph = max(1, self.prev_bbox[3] - self.prev_bbox[1])
            h_loss = (ph - h_b) / ph
            down = (y2 - self.prev_bbox[3]) / ph
            vel = abs(down) / dt
            self.velocity_hist.append(vel)
            avg_vel = float(np.mean(self.velocity_hist))
            
            if h_loss > HEIGHT_LOSS_FALL and avg_vel > VELOCITY_FALL_MIN:
                motion_score = min(1.0, h_loss * 1.5 + avg_vel)
                scores.append(motion_score * 0.15)

        if cls_id == 0 and cls_conf > 0.40:
            scores.append(cls_conf * 0.10)
        elif cls_id == 1:
            return angle_score * 0.3
        elif cls_id == 2 and cls_conf > 0.35:
            return 0.0

        raw = min(1.0, sum(scores))
        self.fall_hist.append(raw)
        
        return float(np.mean(self.fall_hist)) if len(self.fall_hist) >= 2 else raw

    def _stand_conf_bbox(self, bbox, cls_id, cls_conf):
        x1, y1, x2, y2 = bbox
        w_b = max(1, x2 - x1)
        h_b = max(1, y2 - y1)
        scores = []
        ar = h_b / w_b
        
        angle = self._calculate_body_angle_from_bbox(bbox)
        if angle < ANGLE_STAND_MAX:
            angle_stand = 1.0
        elif angle < ANGLE_SITTING_MAX:
            angle_stand = 1.0 - (angle - ANGLE_STAND_MAX) / (ANGLE_SITTING_MAX - ANGLE_STAND_MAX)
        else:
            angle_stand = 0.0
        scores.append(angle_stand * 0.50)
        
        if ar > AR_STAND_MIN:
            scores.append(1.0 * 0.30)
        elif ar > 1.40:
            scores.append((ar - 1.40) / 0.40 * 0.30)
        
        if cls_id == 2 and cls_conf > 0.35:
            scores.append(cls_conf * 0.20)
        
        raw = float(np.mean(scores)) if scores else 0.0
        self.stand_hist.append(raw)
        return float(np.mean(self.stand_hist)) if len(self.stand_hist) >= 2 else raw

    def _fall_conf_pose(self, keypoints, frame_shape):
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        
        kps = keypoints[0]
        fh, fw = frame_shape[:2]
        scores = []
        
        if len(kps) >= 13:
            ls, rs, lh, rh = kps[5], kps[6], kps[11], kps[12]
            if all(k[2] > 0.25 for k in [ls, rs, lh, rh]):
                sh_c = (ls[:2] + rs[:2]) / 2
                hp_c = (lh[:2] + rh[:2]) / 2
                dx = hp_c[0] - sh_c[0]
                dy = hp_c[1] - sh_c[1]
                angle = math.degrees(math.atan2(abs(dx), max(1e-4, abs(dy))))
                self.angle_hist.append(angle)
                avg_a = float(np.mean(self.angle_hist))
                
                if avg_a >= ANGLE_FALL_HIGH:
                    angle_score = 1.0
                elif avg_a >= ANGLE_FALL_MIN:
                    angle_score = (avg_a - ANGLE_FALL_MIN) / (ANGLE_FALL_HIGH - ANGLE_FALL_MIN) * 0.8 + 0.2
                elif avg_a > ANGLE_SITTING_MAX:
                    angle_score = (avg_a - ANGLE_SITTING_MAX) / (ANGLE_FALL_MIN - ANGLE_SITTING_MAX) * 0.2
                else:
                    angle_score = 0.0
                scores.append(angle_score * 0.45)
        
        valid = [k for k in kps if k[2] > 0.25]
        if len(valid) >= 4:
            ys = [k[1] for k in valid]
            xs = [k[0] for k in valid]
            kh = max(ys) - min(ys) + 1e-4
            kw = max(xs) - min(xs) + 1e-4
            ar = kh / kw
            if ar < 1.20:
                ar_score = 1.0
            elif ar < AR_FALL_ZONE_MAX:
                ar_score = 1.0 - (ar - 1.20) / 0.45
            else:
                ar_score = 0.0
            scores.append(ar_score * 0.25)
        
        if len(kps) >= 17:
            ankles = [kps[i] for i in [15, 16] if kps[i][2] > 0.25]
            heads = [kps[i] for i in [3, 4] if kps[i][2] > 0.25]
            if ankles and heads:
                ank_y = max(k[1] for k in ankles)
                hd_y = min(k[1] for k in heads)
                if ank_y > hd_y:
                    ground_score = min(1.0, (ank_y / fh) * 1.25)
                    scores.append(ground_score * 0.30)
        
        raw = min(1.0, sum(scores))
        self.fall_hist.append(raw)
        return float(np.mean(self.fall_hist)) if len(self.fall_hist) >= 2 else raw

    def _stand_conf_pose(self, keypoints):
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        
        kps = keypoints[0]
        scores = []
        
        if len(kps) >= 13:
            ls, rs, lh, rh = kps[5], kps[6], kps[11], kps[12]
            if all(k[2] > 0.25 for k in [ls, rs, lh, rh]):
                sh_c = (ls[:2] + rs[:2]) / 2
                hp_c = (lh[:2] + rh[:2]) / 2
                dx = hp_c[0] - sh_c[0]
                dy = hp_c[1] - sh_c[1]
                angle = math.degrees(math.atan2(abs(dx), max(1e-4, abs(dy))))
                
                if angle < ANGLE_STAND_MAX:
                    scores.append(1.0 * 0.70)
                elif angle < ANGLE_SITTING_MAX:
                    scores.append((1.0 - (angle - ANGLE_STAND_MAX) / (ANGLE_SITTING_MAX - ANGLE_STAND_MAX)) * 0.70)
        
        valid = [k for k in kps if k[2] > 0.25]
        if len(valid) >= 4:
            ys = [k[1] for k in valid]
            scores.append(min(1.0, (max(ys) - min(ys)) / 320) * 0.30)
        
        raw = float(np.mean(scores)) if scores else 0.0
        self.stand_hist.append(raw)
        return float(np.mean(self.stand_hist)) if len(self.stand_hist) >= 2 else raw

    def _update_state(self, fall_conf, stand_conf, frame):
        now = time.time()
        
        if self.state == "MONITORING":
            if fall_conf > FALL_CONF_THRESH:
                self.consecutive_fall_frms += 1
            else:
                self.consecutive_fall_frms = max(0, self.consecutive_fall_frms - 1)
            
            if self.consecutive_fall_frms >= REQUIRED_FALL_FRM:
                self.state = "FALL_DETECTED"
                self.fall_start_time = now
                self.total_falls += 1
                self.consecutive_stand_frms = 0
                
                # ── START RECORDING ON FALL DETECTED ──────────────────────────
                self.recorder.start(frame)
                
                if now - self.last_alert_time > ALERT_COOLDOWN:
                    self.last_alert_time = now
                    msg = f"🚨 Fall detected with {fall_conf:.1%} confidence!"
                    alert_queue.put(("fall", msg, float(fall_conf)))
                    play_alarm_sound("fall")
                    print(f"\n🔴 FALL DETECTED! Conf={fall_conf:.2%} | Angle={int(self.body_angle)}° | Total={self.total_falls}")

        elif self.state == "FALL_DETECTED":
            # ── WRITE FRAME TO RECORDING ────────────────────────────────────
            self.recorder.write(frame)

            if stand_conf > 0.60:
                self.consecutive_stand_frms += 1
            else:
                self.consecutive_stand_frms = max(0, self.consecutive_stand_frms - 1)
            
            if self.consecutive_stand_frms >= REQUIRED_STAND_FRM:
                self._clear_fall("   ✅ Person stood up — resuming monitoring.")
            elif now - self.fall_start_time > 30:
                self._clear_fall("   🔄 Fall timeout — resuming monitoring.")

    def _clear_fall(self, msg):
        # ── STOP RECORDING WHEN FALL ENDS ───────────────────────────────────
        self.recorder.stop()
        self.state = "MONITORING"
        self.consecutive_fall_frms = 0
        self.consecutive_stand_frms = 0
        self.fall_hist.clear()
        print(msg)

    def process_frame(self, frame):
        proc = cv2.resize(frame, (FRAME_W, FRAME_H))
        results = self.model(proc, verbose=False, conf=0.30, imgsz=640)
        
        fall_conf = stand_conf = 0.0
        keypoints = bbox = None
        cls_id = None
        cls_conf = 0.0
        
        if self.use_custom:
            if results and results[0].boxes is not None:
                best = 0.0
                for box in results[0].boxes:
                    c = float(box.conf[0])
                    ci = int(box.cls[0])
                    if c > best:
                        best = c
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox = (x1, y1, x2, y2)
                        cls_id = ci
                        cls_conf = c
                if bbox:
                    fall_conf = self._fall_conf_bbox(bbox, cls_id, cls_conf)
                    stand_conf = self._stand_conf_bbox(bbox, cls_id, cls_conf)
        else:
            if results and results[0].keypoints is not None:
                kps = results[0].keypoints.data.cpu().numpy()
                if len(kps) > 0:
                    keypoints = kps
                    fall_conf = self._fall_conf_pose(keypoints, proc.shape)
                    stand_conf = self._stand_conf_pose(keypoints)
        
        if bbox is not None:
            self.prev_bbox = bbox
            self.prev_time = time.time()
        
        self._update_state(fall_conf, stand_conf, proc)

        # ── DATASET COLLECTION (fall only) ─────────────────────────────────
        self.dataset.try_save(
            frame          = proc,
            detector_state = self.state,
            fall_conf      = fall_conf,
            stand_conf     = stand_conf,
            bbox           = bbox,
            cls_id         = cls_id,
            cls_conf       = cls_conf,
            body_angle     = self.body_angle,
        )
        
        self.shared_state['fall'] = {
            'state':      self.state,
            'confidence': fall_conf,
            'total':      self.total_falls,
            'model':      self.model_label,
            'angle':      int(self.body_angle) if hasattr(self, 'body_angle') else 0,
            'ds_fallen':   self.dataset._total_saved[0],
            'ds_sitting':  self.dataset._total_saved[1],
            'ds_standing': self.dataset._total_saved[2],
            'rec_active':  self.recorder.is_recording,
            'rec_clips':   self.recorder._total_clips,
        }
        
        return fall_conf, stand_conf, keypoints, bbox, cls_id, cls_conf

    def fall_detection_loop(self):
        cap = None
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0
        
        for idx in range(3):
            try:
                c = (cv2.VideoCapture(idx, backend) if platform.system() == "Windows"
                     else cv2.VideoCapture(idx))
                if c.isOpened():
                    ret, frm = c.read()
                    if ret and frm is not None:
                        cap = c
                        print(f"   ✅ Fall camera at index {idx}")
                        break
                c.release()
            except Exception:
                pass
        
        if cap is None:
            print("   ❌ Cannot open camera")
            self.running = False
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        err = 0
        
        while self.running:
            try:
                ret, frame = cap.read()
                if not ret or frame is None:
                    err += 1
                    if err > 10:
                        print("   ⚠️ Camera lost")
                        break
                    time.sleep(0.1)
                    continue
                err = 0
                self.process_frame(frame)
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
            except Exception as e:
                print(f"   ⚠️ Fall error: {e}")
                time.sleep(0.5)
        
        if self.recorder.is_recording:
            self.recorder.stop()

        cap.release()
        print("   👋 Fall detection stopped")

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE DETECTOR — COMPLETELY UNCHANGED
# ══════════════════════════════════════════════════════════════════════════════
class UnifiedVoiceDetector:
    def __init__(self, alert_sender, shared_state):
        print("   🎤 Initializing Voice Detection...")
        self.alert_sender     = alert_sender
        self.shared_state     = shared_state
        self.emergency_count  = 0
        self.last_alert_time  = 0
        self.alert_cooldown   = 15
        self.running          = True
        self.listening_status = "Initializing"

        self.keywords = {
            'english':  ['help', 'emergency', 'accident', 'fall', 'fell',
                         'hurt', 'pain', 'save', 'please help', 'help me'],
            'malayalam':['സഹായം','അടിയന്തരം','അപകടം','വീഴ്ച','വീണു','വേദന'],
            'hindi':    ['मदद','आपातकाल','दुर्घटना','गिर गया','चोट','दर्द'],
        }
        self.supported_languages = ['en-IN', 'ml-IN', 'hi-IN']
        self.current_language    = 'en-IN'

        self.recognizer = sr.Recognizer()

        MIC_INDEX = 2

        try:
            mic_names = sr.Microphone.list_microphone_names()
            if MIC_INDEX is not None and MIC_INDEX < len(mic_names):
                mic_name = mic_names[MIC_INDEX]
                print(f"   🎤 Mic: [{MIC_INDEX}] {mic_name}")
            else:
                if MIC_INDEX is not None:
                    print(f"   ⚠️  Mic index {MIC_INDEX} not found — using system default")
                MIC_INDEX = None
                print("   🎤 Mic: system default")
        except Exception:
            MIC_INDEX = None
            print("   🎤 Mic: system default (could not read device list)")

        self.mic_index  = MIC_INDEX
        self.microphone = sr.Microphone(device_index=MIC_INDEX)

        print("   🔊 Calibrating microphone...")
        try:
            with self.microphone as source:
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.energy_threshold         = 300
                self.recognizer.adjust_for_ambient_noise(source, duration=2)
                print(f"   ✅ Calibrated (threshold: {self.recognizer.energy_threshold:.0f})")
        except Exception as e:
            print(f"   ⚠️  Calibration failed: {e}")
            print("      → Connect headset before starting, or set MIC_INDEX = None")

        self.listening_status = "Listening"
        print("   ✅ Voice Detection Ready")

    def detect_language(self, text):
        if re.search(r'[\u0D00-\u0D7F]', text): return 'malayalam'
        if re.search(r'[\u0900-\u097F]', text): return 'hindi'
        return 'english'

    def check_emergency_keywords(self, text):
        text_lower = text.lower()
        found = []
        for lang, words in self.keywords.items():
            for word in words:
                if word.lower() in text_lower:
                    found.append(word)
        return found

    def transcribe_speech(self, audio):
        try:
            text = self.recognizer.recognize_google(
                audio, language=self.current_language)
            return text, True
        except sr.UnknownValueError:
            pass
        except sr.RequestError as e:
            print(f"   ⚠️  Google Speech API error: {e}")
            print("         Check internet connection.")
            return None, False
        except Exception as e:
            print(f"   ⚠️  Transcription error: {e}")
            return None, False

        for lang in self.supported_languages:
            if lang == self.current_language:
                continue
            try:
                text = self.recognizer.recognize_google(audio, language=lang)
                return text, True
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"   ⚠️  API error ({lang}): {e}")
                break
            except Exception:
                continue

        return None, False

    def voice_detection_loop(self):
        print("\n   🎤 Voice detection active — listening for keywords")
        print(f"   ℹ️  Keywords: {', '.join(self.keywords['english'])}")

        while self.running:
            try:
                mic = sr.Microphone(device_index=self.mic_index)
            except Exception as e:
                print(f"   ⚠️  Cannot open mic (index {self.mic_index}): {e}")
                print("      → Is your headset connected? Retrying in 5s...")
                time.sleep(5)
                continue

            try:
                with mic as source:
                    while self.running:
                        try:
                            audio = self.recognizer.listen(
                                source,
                                timeout=3,
                                phrase_time_limit=6
                            )
                            self.listening_status = "Processing..."

                        except sr.WaitTimeoutError:
                            self.listening_status = "Listening"
                            self._update_shared_state()
                            continue

                        text, success = self.transcribe_speech(audio)

                        if success and text:
                            print(f"\n   🗣️  Heard: \"{text}\"")
                            self.listening_status = f"Heard: {text[:25]}"
                            keywords = self.check_emergency_keywords(text)

                            if keywords:
                                now = time.time()
                                if now - self.last_alert_time > self.alert_cooldown:
                                    self.last_alert_time  = now
                                    self.emergency_count += 1
                                    message = f"🚨 Voice emergency! Keywords: {', '.join(keywords)}"
                                    alert_queue.put(("voice", message))
                                    print(f"\n   🚨 VOICE EMERGENCY! Keywords={keywords} | Total={self.emergency_count}")
                                    play_alarm_sound("voice")
                                else:
                                    remaining = self.alert_cooldown - (now - self.last_alert_time)
                                    print(f"   ⏳ Cooldown {remaining:.0f}s remaining")
                            else:
                                print(f"   ℹ️  No emergency keywords in: \"{text}\"")

                        self.listening_status = "Listening"
                        self._update_shared_state()

            except Exception as e:
                if self.running:
                    print(f"   ⚠️  Mic error: {e}")
                    print("      → Reconnect headset or change MIC_INDEX. Retrying in 3s...")
                    self.listening_status = "Mic error — retrying"
                    self._update_shared_state()
                    time.sleep(3)

    def _update_shared_state(self):
        self.shared_state['voice'] = {
            'status': self.listening_status,
            'total':  self.emergency_count,
        }

    def stop(self):
        self.running = False


# ══════════════════════════════════════════════════════════════════════════════
#  ALERT HANDLER THREAD
# ══════════════════════════════════════════════════════════════════════════════
def alert_handler(alert_sender):
    print("\n📡 Alert handler started")
    while True:
        try:
            item = alert_queue.get(timeout=1)
            if len(item) == 3:
                alert_type, message, confidence = item
                print(f"\n📱 Sending {alert_type} alert  conf={confidence:.2%}")
                alert_sender.send_alert(alert_type, message, confidence)
            elif len(item) == 2:
                alert_type, message = item
                print(f"\n📱 Sending {alert_type} alert")
                alert_sender.send_alert(alert_type, message)
            else:
                print(f"❌ Unknown alert format: {item}")
        except queue.Empty:
            continue
        except Exception as e:
            print(f"❌ Alert handler error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY THREAD
# ══════════════════════════════════════════════════════════════════════════════
def display_thread(fall_detector, shared_state):
    cv2.namedWindow("Guardian Net - Fall & Voice Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Guardian Net - Fall & Voice Detection", 800, 600)

    shared_state.setdefault('voice', {'status': 'Listening', 'total': 0})
    shared_state.setdefault('fall',  {
        'state': 'MONITORING', 'confidence': 0,
        'total': 0, 'model': '', 'angle': 0,
        'ds_fallen': 0, 'ds_sitting': 0, 'ds_standing': 0,
        'rec_active': False, 'rec_clips': 0,
    })

    fps_start = time.time()
    fps_count = 0
    fps = 0

    while fall_detector.running:
        try:
            frame = fall_detector.frame_queue.get(timeout=1)
            disp = frame.copy()
            h, w = disp.shape[:2]

            fps_count += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_start = time.time()

            fall_state = shared_state.get('fall', {})
            voice_state = shared_state.get('voice', {})
            is_fall = fall_state.get('state') == "FALL_DETECTED"
            fall_conf = float(fall_state.get('confidence', 0))
            body_angle = fall_state.get('angle', 0)

            ov = disp.copy()
            cv2.rectangle(ov, (0, 0), (w, 90), (10, 10, 10), -1)
            cv2.addWeighted(ov, 0.75, disp, 0.25, 0, disp)
            cv2.putText(disp, "GUARDIAN NET", (w//2 - 105, 30),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (80, 220, 80), 2)
            cv2.putText(disp, "FALL & VOICE DETECTION", (w//2 - 130, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            cv2.putText(disp, f"{fps} FPS", (w - 85, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 80), 1)

            mdl = fall_state.get('model', '')
            is_cu = mdl not in ('', 'POSE')
            cv2.putText(disp, f"[{mdl}]", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (60, 220, 60) if is_cu else (0, 200, 220), 1)
            cv2.putText(disp, "CUSTOM" if is_cu else "POSE", (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        (60, 220, 60) if is_cu else (0, 200, 220), 1)

            cv2.putText(disp, f"ANGLE: {body_angle}°", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (0, 200, 255) if body_angle > 35 else (100, 200, 100), 1)

            cy, ch = 98, 65
            pulse = int(time.time() * 4) % 2
            if is_fall:
                bg = (20, 0, 0) if pulse else (40, 0, 0)
                bdr = (0, 60, 255)
                txt = "FALL DETECTED!"
                sub = "EMERGENCY ALERT SENT"
                tc = (0, 80, 255)
            else:
                bg = (0, 28, 0)
                bdr = (0, 180, 0)
                txt = "MONITORING"
                sub = "No fall detected"
                tc = (80, 240, 80)
            cv2.rectangle(disp, (30, cy), (w - 30, cy + ch), bg, -1)
            cv2.rectangle(disp, (30, cy), (w - 30, cy + ch), bdr, 2)
            cv2.putText(disp, txt, (w//2 - 110, cy + 28), cv2.FONT_HERSHEY_DUPLEX, 0.80, tc, 2)
            cv2.putText(disp, sub, (w//2 - 130, cy + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.46, tc, 1)

            gy = cy + ch + 14
            gw = w - 70
            cv2.rectangle(disp, (35, gy), (35 + gw, gy + 20), (35, 35, 35), -1)
            cv2.rectangle(disp, (35, gy), (35 + gw, gy + 20), (70, 70, 70), 1)
            fill = int(gw * min(1.0, fall_conf))
            if fill > 0:
                cv2.rectangle(disp, (35, gy), (35 + fill, gy + 20),
                              (0, 70, 255) if is_fall else (0, 200, 0), -1)
            tx = 35 + int(gw * FALL_CONF_THRESH)
            cv2.line(disp, (tx, gy - 4), (tx, gy + 24), (255, 165, 0), 2)
            cv2.putText(disp, f"FALL CONF: {fall_conf*100:.0f}%  [thresh {FALL_CONF_THRESH*100:.0f}%]",
                        (35, gy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (210, 210, 210), 1)

            vy = gy + 20 + 14
            v_status = voice_state.get('status', 'Listening')
            if "EMERGENCY" in v_status or "emergency" in v_status.lower():
                vc = (0, 60, 255)
            elif "Heard" in v_status:
                vc = (0, 220, 255)
            elif "Processing" in v_status:
                vc = (0, 255, 180)
            elif "Error" in v_status:
                vc = (0, 100, 255)
            else:
                vc = (0, 160, 200)
            cv2.putText(disp, f"MIC: {v_status}", (35, vy + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, vc, 1)

            sy = vy + 28
            cv2.rectangle(disp, (20, sy), (w - 20, sy + 48), (12, 12, 12), -1)
            cols = [
                (f"Patient: {fall_detector.alert_sender.patient_id}", 30),
                (f"Falls: {fall_state.get('total', 0)}", 30 + w//4),
                (f"Voice alerts: {voice_state.get('total', 0)}", 30 + w//2),
                (f"Sent: {fall_detector.alert_sender.alert_count}", 30 + 3*w//4),
            ]
            for t2, xp in cols:
                c2 = ((0, 80, 255) if "Falls" in t2 and fall_state.get('total', 0) > 0 else
                      (0, 200, 255) if "Voice" in t2 and voice_state.get('total', 0) > 0 else
                      (185, 185, 185))
                cv2.putText(disp, t2, (xp, sy + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, c2, 1)

            # ── Dataset counter row (fall-only) ────────────────────────────
            dy = sy + 48 + 6
            df = fall_state.get('ds_fallen', 0)
            cv2.rectangle(disp, (20, dy), (w - 20, dy + 22), (8, 8, 20), -1)
            ds_txt = f"💾 Dataset  Fallen:{df}  (sitting/standing not collected)"
            cv2.putText(disp, ds_txt, (28, dy + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 200, 255), 1)

            # ── Recording indicator row ────────────────────────────────────
            ry = dy + 22 + 4
            rec_active = fall_state.get('rec_active', False)
            rec_clips  = fall_state.get('rec_clips',  0)
            cv2.rectangle(disp, (20, ry), (w - 20, ry + 22), (8, 8, 8), -1)
            if rec_active:
                blink = int(time.time() * 2) % 2
                dot_col = (0, 0, 220) if blink else (0, 0, 160)
                cv2.circle(disp, (36, ry + 11), 6, dot_col, -1)
                rec_txt = f"● REC (max 10s)  Clips saved: {rec_clips}"
                rc = (0, 80, 255)
            else:
                rec_txt = f"○ NOT RECORDING  Clips saved: {rec_clips}"
                rc = (120, 120, 120)
            cv2.putText(disp, rec_txt, (48, ry + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, rc, 1)

            cv2.putText(disp, "Q=quit", (w - 65, h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)

            cv2.imshow("Guardian Net - Fall & Voice Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                fall_detector.running = False
                break

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Display error: {e}")
            break

    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Guardian Net Detector")
    parser.add_argument("--patient_id", type=int, default=1,
                        help="Patient ID to monitor (default: 1)")
    args, _ = parser.parse_known_args()
    patient_id = args.patient_id

    print("\n" + "="*70)
    print("🚀 GUARDIAN NET — ENHANCED FALL + VOICE DETECTION")
    print("="*70)
    print(f"\n📱 Patient ID: {patient_id}")
    print("="*70)

    alert_sender = GuardianAlertSender(patient_id=patient_id)
    if hasattr(alert_sender, 'test_connection'):
        if alert_sender.test_connection():
            print("✅ Connected to Guardian Net server")
        else:
            print("⚠️  Cannot connect — alerts logged only")

    shared_state = {
        'voice': {'status': 'Starting...', 'total': 0},
        'fall':  {
            'state': 'Starting...', 'confidence': 0, 'total': 0,
            'model': '', 'angle': 0,
            'ds_fallen': 0, 'ds_sitting': 0, 'ds_standing': 0,
            'rec_active': False, 'rec_clips': 0,
        },
    }

    print("\n🔧 Initializing detectors...")
    fall_detector = UnifiedFallDetector(alert_sender, shared_state)
    voice_detector = UnifiedVoiceDetector(alert_sender, shared_state)

    print("\n" + "="*70)
    print("✅ ALL DETECTORS READY — Starting threads...")
    print("="*70)
    print("📹 Fall  : Camera (Improved Angle Detection)")
    print(f"   ⚙️  Confidence threshold: {FALL_CONF_THRESH} | Angle threshold: {ANGLE_FALL_MIN}°")
    print("🎤 Voice : Microphone  (EN / Malayalam / Hindi)")
    print("💾 Dataset: realtime_dataset/  (fall frames only)")
    print(f"🎥 Video : realtime_dataset/fall_recordings/  (max {FALL_RECORD_MAX_FRAMES/15:.0f}s clips)")
    print("💡 RETRAIN: After 50+ samples, retrain command will appear in terminal")
    print("\nPress Q in the video window to quit")
    print("="*70 + "\n")

    threads = [
        threading.Thread(target=fall_detector.fall_detection_loop,   daemon=True, name="Fall"),
        threading.Thread(target=voice_detector.voice_detection_loop,  daemon=True, name="Voice"),
        threading.Thread(target=alert_handler, args=(alert_sender,),  daemon=True, name="Alert"),
        threading.Thread(target=display_thread,
                         args=(fall_detector, shared_state),          daemon=True, name="Display"),
    ]
    for t in threads:
        t.start()

    try:
        while fall_detector.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    fall_detector.stop()
    voice_detector.stop()

    fall_detector.dataset.print_summary()
    fall_detector.recorder.print_summary()

    print("\n📊 Final Summary")
    print("="*50)
    print(f"   Falls detected   : {fall_detector.total_falls}")
    print(f"   Voice emergencies: {voice_detector.emergency_count}")
    print(f"   Alerts sent      : {alert_sender.alert_count}")
    print("="*50)
    print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()