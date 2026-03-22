"""
GUARDIAN NET - ENHANCED FALL DETECTION SYSTEM
Optimized for human_fall_detection custom dataset
Supports: Custom YOLO model + Pose fallback
Features: Multi-feature fusion, adaptive thresholding, cross-platform alarm
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque
import warnings
import os
import sys
import threading
import math
import platform

# ─── Cross-platform alarm ──────────────────────────────────────────────────────
def play_alarm_sound():
    """Cross-platform emergency alarm using beeps"""
    system = platform.system()

    def _play():
        if system == "Windows":
            import winsound
            pattern = [(1200, 180), (900, 180), (1200, 180), (900, 180), (1500, 500)]
            for freq, dur in pattern:
                winsound.Beep(freq, dur)
                time.sleep(0.05)
        elif system == "Darwin":  # macOS
            os.system('say "Fall detected, emergency alert"')
        else:  # Linux / fallback
            try:
                os.system('paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga 2>/dev/null || '
                          'aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || '
                          'python3 -c "import subprocess; subprocess.run([\'beep\', \'-f\', \'1000\', \'-l\', \'200\'])"')
            except Exception:
                pass  # Silent fallback

    threading.Thread(target=_play, daemon=True).start()

# ─── Try importing GuardianAlertSender ────────────────────────────────────────
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from guardian_integration import GuardianAlertSender
    GUARDIAN_AVAILABLE = True
except ImportError:
    GUARDIAN_AVAILABLE = False

    class GuardianAlertSender:
        """Fallback stub when guardian_integration is missing"""
        def __init__(self, patient_id=1):
            self.patient_id = patient_id
            self.alert_count = 0
            print("⚠️  guardian_integration not found – alerts will be logged only.")

        def send_alert(self, alert_type, message, confidence):
            self.alert_count += 1
            ts = time.strftime("%H:%M:%S")
            print(f"[ALERT {ts}] Type={alert_type} | Conf={confidence:.2%} | {message}")

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES       = ["fallen", "sitting", "standing"]
FRAME_W, FRAME_H  = 640, 480

# ── Detection thresholds ──────────────────────────────────────────────────────
FALL_CONF_THRESH   = 0.55     # Min smoothed confidence to count a frame as "fall"
REQUIRED_FALL_FRM  = 5        # Consecutive positive frames before triggering
REQUIRED_STAND_FRM = 8        # Consecutive standing frames to clear a fall
ALERT_COOLDOWN     = 8        # Seconds between repeated alerts

# ── Aspect-ratio bounds ───────────────────────────────────────────────────────
AR_FALL_MAX        = 1.10     # h/w ratio: below this → definitely horizontal
AR_FALL_ZONE_MAX   = 1.65     # Partial score between these two
AR_STAND_MIN       = 1.80     # h/w ratio: above this → definitely vertical

# ── Body angle bounds (degrees from vertical) ────────────────────────────────
ANGLE_FALL_MIN     = 40       # Shoulder-hip line tilted > 40° → fall cue
ANGLE_STAND_MAX    = 25       # < 25° tilt → standing cue


# ══════════════════════════════════════════════════════════════════════════════
#  FALL DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
class FallDetectorPro:
    def __init__(self, patient_id: int = 1):
        self._print_banner()
        self.patient_id = patient_id
        self.alert_sender = GuardianAlertSender(patient_id=patient_id)

        # ── Load model ────────────────────────────────────────────────────────
        self.model, self.use_custom, self.model_label = self._load_model()

        # ── State machine ─────────────────────────────────────────────────────
        self.state                  = "MONITORING"
        self.total_falls            = 0
        self.consecutive_fall_frms  = 0
        self.consecutive_stand_frms = 0
        self.fall_start_time        = 0.0
        self.last_alert_time        = 0.0
        self.alarm_active           = False

        # ── Smoothing histories ───────────────────────────────────────────────
        self.fall_hist      = deque(maxlen=8)   # Smoothed fall confidence
        self.stand_hist     = deque(maxlen=8)
        self.angle_hist     = deque(maxlen=6)
        self.velocity_hist  = deque(maxlen=8)
        self.height_hist    = deque(maxlen=20)

        # ── Motion state ──────────────────────────────────────────────────────
        self.prev_bbox      = None
        self.prev_time      = None

        # ── FPS ───────────────────────────────────────────────────────────────
        self.fps_start  = time.time()
        self.fps_count  = 0
        self.fps        = 0

        # ── Stats ─────────────────────────────────────────────────────────────
        self.total_frames     = 0
        self.detection_frames = 0

        print(f"\n✅ READY — {self.model_label}")
        print(f"   Threshold  : {FALL_CONF_THRESH}")
        print(f"   Trigger    : {REQUIRED_FALL_FRM} consecutive frames")
        print(f"   Cooldown   : {ALERT_COOLDOWN}s")
        print("=" * 78 + "\n")

    # ── Init helpers ──────────────────────────────────────────────────────────
    @staticmethod
    def _print_banner():
        print("\n" + "=" * 78)
        print("  🏥  GUARDIAN NET — ENHANCED FALL DETECTION PRO")
        print("=" * 78)

    def _load_model(self):
        base = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(base)

        candidates = [
            os.path.join(root, "runs", "train", "fall_custom_scratch", "weights", "best.pt"),
            os.path.join(root, "runs", "train", "fall_detection",      "weights", "best.pt"),
            os.path.join(base, "best.pt"),
        ]

        for p in candidates:
            if os.path.exists(p):
                print(f"✅ Custom model → {p}")
                return YOLO(p), True, "CUSTOM"

        # Fallback: pose model
        pose_path = os.path.join(base, "yolov8n-pose.pt")
        if not os.path.exists(pose_path):
            print("📥 Downloading yolov8n-pose.pt …")
        return YOLO("yolov8n-pose.pt"), False, "POSE"

    # ══════════════════════════════════════════════════════════════════════════
    #  CONFIDENCE CALCULATORS
    # ══════════════════════════════════════════════════════════════════════════

    # ── CUSTOM MODEL (bounding-box path) ──────────────────────────────────────
    def _fall_conf_bbox(self, bbox, cls_id, cls_conf) -> float:
        x1, y1, x2, y2 = bbox
        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)
        scores = []

        # 1. Aspect ratio  (weight 0.35)
        ar = h_box / w_box
        if ar < AR_FALL_MAX:
            ar_score = 1.0
        elif ar < AR_FALL_ZONE_MAX:
            ar_score = 1.0 - (ar - AR_FALL_MAX) / (AR_FALL_ZONE_MAX - AR_FALL_MAX)
        else:
            ar_score = 0.0
        scores.append(ar_score * 0.35)

        # 2. Ground proximity (weight 0.25)
        center_y_norm = ((y1 + y2) / 2) / FRAME_H
        if center_y_norm > 0.55:
            scores.append(min(1.0, (center_y_norm - 0.45) * 2.2) * 0.25)

        # 3. Downward velocity (weight 0.25)
        if self.prev_bbox is not None and self.prev_time is not None:
            dt = max(0.02, time.time() - self.prev_time)
            py2 = self.prev_bbox[3]
            ph  = max(1, self.prev_bbox[3] - self.prev_bbox[1])
            h_loss = (ph - h_box) / ph
            down   = (y2 - py2) / ph
            vel    = abs(down) / dt
            self.velocity_hist.append(vel)
            avg_vel = float(np.mean(self.velocity_hist))
            if h_loss > 0.12 and avg_vel > 0.10:
                scores.append(min(1.0, h_loss * 1.5 + avg_vel) * 0.25)

        # 4. Model confidence for "fallen" class (weight 0.15)
        if cls_id == 0 and cls_conf > 0.35:
            scores.append(cls_conf * 0.15)

        raw = min(1.0, sum(scores))
        self.fall_hist.append(raw)
        return float(np.mean(self.fall_hist)) if len(self.fall_hist) >= 2 else raw

    def _stand_conf_bbox(self, bbox, cls_id, cls_conf) -> float:
        x1, y1, x2, y2 = bbox
        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)
        scores = []

        ar = h_box / w_box
        if ar > AR_STAND_MIN:
            scores.append(min(1.0, (ar - AR_STAND_MIN) / 0.6 + 0.4) * 0.70)

        if cls_id == 2 and cls_conf > 0.35:
            scores.append(cls_conf * 0.30)

        raw = float(np.mean(scores)) if scores else 0.0
        self.stand_hist.append(raw)
        return float(np.mean(self.stand_hist)) if len(self.stand_hist) >= 2 else raw

    # ── POSE MODEL (keypoints path) ───────────────────────────────────────────
    def _fall_conf_pose(self, keypoints, frame_shape) -> float:
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        kps = keypoints[0]
        fh, fw = frame_shape[:2]
        scores = []

        # 1. Shoulder–hip angle (weight 0.40)
        if len(kps) >= 13:
            ls, rs = kps[5], kps[6]
            lh, rh = kps[11], kps[12]
            if all(k[2] > 0.3 for k in [ls, rs, lh, rh]):
                sh_c = (ls[:2] + rs[:2]) / 2
                hp_c = (lh[:2] + rh[:2]) / 2
                dx, dy = hp_c[0] - sh_c[0], hp_c[1] - sh_c[1]
                angle = math.degrees(math.atan2(abs(dx), max(1e-4, abs(dy))))
                self.angle_hist.append(angle)
                avg_a = float(np.mean(self.angle_hist))
                if avg_a > ANGLE_FALL_MIN:
                    scores.append(min(1.0, (avg_a - 30) / 55) * 0.40)

        # 2. Bounding aspect ratio of visible keypoints (weight 0.30)
        valid = [k for k in kps if k[2] > 0.25]
        if len(valid) >= 4:
            ys = [k[1] for k in valid]
            xs = [k[0] for k in valid]
            kp_h = max(ys) - min(ys) + 1e-4
            kp_w = max(xs) - min(xs) + 1e-4
            ar = kp_h / kp_w
            if ar < 1.20:
                scores.append((1.0 - ar / 1.20) * 0.30)
            elif ar < AR_FALL_ZONE_MAX:
                scores.append((1.0 - (ar - 1.20) / 0.45) * 0.15)

        # 3. Ground proximity (weight 0.30)
        if len(kps) >= 17:
            ankles = [kps[i] for i in [15, 16] if kps[i][2] > 0.25]
            heads  = [kps[i] for i in [3, 4]   if kps[i][2] > 0.25]
            if ankles and heads:
                ank_y = max(k[1] for k in ankles)
                hd_y  = min(k[1] for k in heads)
                if ank_y > hd_y:
                    scores.append(min(1.0, (ank_y / fh) * 1.25) * 0.30)

        raw = min(1.0, sum(scores))
        self.fall_hist.append(raw)
        return float(np.mean(self.fall_hist)) if len(self.fall_hist) >= 2 else raw

    def _stand_conf_pose(self, keypoints) -> float:
        if keypoints is None or len(keypoints) == 0:
            return 0.0
        kps = keypoints[0]
        scores = []

        if len(kps) >= 13:
            ls, rs = kps[5], kps[6]
            lh, rh = kps[11], kps[12]
            if all(k[2] > 0.3 for k in [ls, rs, lh, rh]):
                sh_c = (ls[:2] + rs[:2]) / 2
                hp_c = (lh[:2] + rh[:2]) / 2
                dx, dy = hp_c[0] - sh_c[0], hp_c[1] - sh_c[1]
                angle = math.degrees(math.atan2(abs(dx), max(1e-4, abs(dy))))
                if angle < ANGLE_STAND_MAX:
                    scores.append(1.0 * 0.70)
                elif angle < 40:
                    scores.append((1.0 - (angle - ANGLE_STAND_MAX) / 15) * 0.70)

        valid = [k for k in kps if k[2] > 0.25]
        if len(valid) >= 4:
            ys   = [k[1] for k in valid]
            span = max(ys) - min(ys)
            scores.append(min(1.0, span / 320) * 0.30)

        raw = float(np.mean(scores)) if scores else 0.0
        self.stand_hist.append(raw)
        return float(np.mean(self.stand_hist)) if len(self.stand_hist) >= 2 else raw

    # ══════════════════════════════════════════════════════════════════════════
    #  STATE MACHINE
    # ══════════════════════════════════════════════════════════════════════════
    def _update_state(self, fall_conf: float, stand_conf: float):
        now = time.time()

        if self.state == "MONITORING":
            if fall_conf > FALL_CONF_THRESH:
                self.consecutive_fall_frms += 1
            else:
                # Soft decay so a single bad frame doesn't reset the counter
                self.consecutive_fall_frms = max(0, self.consecutive_fall_frms - 1)

            if self.consecutive_fall_frms >= REQUIRED_FALL_FRM:
                self.state = "FALL_DETECTED"
                self.fall_start_time = now
                self.total_falls += 1
                self.consecutive_stand_frms = 0

                if now - self.last_alert_time > ALERT_COOLDOWN:
                    self.last_alert_time = now
                    msg = f"🚨 FALL DETECTED — confidence {fall_conf:.1%}"
                    self.alert_sender.send_alert("fall", msg, fall_conf)
                    play_alarm_sound()
                    self._print_fall_alert(fall_conf)

        elif self.state == "FALL_DETECTED":
            if stand_conf > 0.60:
                self.consecutive_stand_frms += 1
            else:
                self.consecutive_stand_frms = max(0, self.consecutive_stand_frms - 1)

            if self.consecutive_stand_frms >= REQUIRED_STAND_FRM:
                self._clear_fall("✅ Person stood up — resuming monitoring.")
            elif now - self.fall_start_time > 30:
                self._clear_fall("🔄 Fall timeout — resuming monitoring.")

    def _clear_fall(self, msg: str):
        self.state = "MONITORING"
        self.consecutive_fall_frms  = 0
        self.consecutive_stand_frms = 0
        self.fall_hist.clear()
        print(msg)

    @staticmethod
    def _print_fall_alert(conf: float):
        print("\n" + "=" * 70)
        print("  🔴🔴🔴  FALL DETECTED — EMERGENCY ALERT SENT  🔴🔴🔴")
        print(f"  Confidence : {conf:.2%}")
        print("=" * 70 + "\n")

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN PROCESS
    # ══════════════════════════════════════════════════════════════════════════
    def process_frame(self, frame):
        self.total_frames += 1
        results = self.model(frame, verbose=False, conf=0.30, imgsz=640)

        fall_conf = stand_conf = 0.0
        keypoints = bbox = None
        cls_id    = None
        cls_conf  = 0.0

        if self.use_custom:
            if results and results[0].boxes is not None:
                best_conf = 0.0
                for box in results[0].boxes:
                    c  = float(box.conf[0])
                    ci = int(box.cls[0])
                    if c > best_conf:
                        best_conf = c
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        bbox    = (x1, y1, x2, y2)
                        cls_id  = ci
                        cls_conf = c

                if bbox:
                    fall_conf  = self._fall_conf_bbox(bbox, cls_id, cls_conf)
                    stand_conf = self._stand_conf_bbox(bbox, cls_id, cls_conf)
                    self.detection_frames += 1
        else:
            if results and results[0].keypoints is not None:
                kps = results[0].keypoints.data.cpu().numpy()
                if len(kps) > 0:
                    keypoints  = kps
                    fall_conf  = self._fall_conf_pose(keypoints, frame.shape)
                    stand_conf = self._stand_conf_pose(keypoints)
                    self.detection_frames += 1

        if bbox is not None:
            self.prev_bbox = bbox
            self.prev_time = time.time()

        self._update_state(fall_conf, stand_conf)
        return fall_conf, stand_conf, keypoints, bbox, cls_id, cls_conf

    # ══════════════════════════════════════════════════════════════════════════
    #  UI RENDERING
    # ══════════════════════════════════════════════════════════════════════════
    def draw_ui(self, frame, fall_conf, stand_conf, keypoints, bbox, cls_id, cls_conf):
        h, w = frame.shape[:2]
        is_fall = (self.state == "FALL_DETECTED")

        # ── Top bar ───────────────────────────────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "GUARDIAN NET", (w // 2 - 105, 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (80, 220, 80), 2)
        cv2.putText(frame, "FALL DETECTION PRO", (w // 2 - 115, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
        # FPS
        cv2.putText(frame, f"{self._calc_fps()} FPS", (w - 90, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 1)
        # Model label
        cv2.putText(frame, self.model_label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)

        # ── Status card ───────────────────────────────────────────────────────
        cy, ch = 108, 72
        pulse = int(time.time() * 4) % 2

        if is_fall:
            bg   = (20, 0, 0) if pulse else (40, 0, 0)
            bdr  = (0, 60, 255)
            txt  = "⚠  FALL DETECTED!"
            sub  = "EMERGENCY ALERT SENT"
            tc   = (0, 80, 255)
        else:
            bg   = (0, 30, 0)
            bdr  = (0, 200, 0)
            txt  = "✔  MONITORING"
            sub  = "System active — no fall detected"
            tc   = (80, 255, 80)

        cv2.rectangle(frame, (35, cy), (w - 35, cy + ch), bg,  -1)
        cv2.rectangle(frame, (35, cy), (w - 35, cy + ch), bdr,  2)
        cv2.putText(frame, txt, (w // 2 - 110, cy + 32),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, tc, 2)
        cv2.putText(frame, sub, (w // 2 - 145, cy + 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, tc, 1)

        # ── Fall confidence gauge ─────────────────────────────────────────────
        gy = cy + ch + 18
        gw = w - 80

        cv2.rectangle(frame, (40, gy), (40 + gw, gy + 22), (35, 35, 35), -1)
        cv2.rectangle(frame, (40, gy), (40 + gw, gy + 22), (75, 75, 75),  1)

        fill = int(gw * min(1.0, fall_conf))
        bar_color = (0, 80, 255) if is_fall else (0, 200, 0)
        if fill > 0:
            cv2.rectangle(frame, (40, gy), (40 + fill, gy + 22), bar_color, -1)

        # Threshold marker
        thresh_x = 40 + int(gw * FALL_CONF_THRESH)
        cv2.line(frame, (thresh_x, gy - 4), (thresh_x, gy + 26), (255, 165, 0), 2)

        cv2.putText(frame, f"FALL CONFIDENCE: {fall_conf * 100:.0f}%  [threshold {FALL_CONF_THRESH * 100:.0f}%]",
                    (40, gy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)

        # ── Metrics strip ─────────────────────────────────────────────────────
        my = gy + 22 + 16
        mh = 58
        cv2.rectangle(frame, (20, my), (w - 20, my + mh), (12, 12, 12), -1)

        det_rate = (self.detection_frames / max(1, self.total_frames)) * 100
        cols = [
            (f"Patient: {self.patient_id}",        30),
            (f"Falls: {self.total_falls}",          30 + w // 4),
            (f"Alerts: {self.alert_sender.alert_count}", 30 + w // 2),
            (f"Det: {det_rate:.0f}%",               30 + 3 * w // 4),
        ]
        for text, xp in cols:
            color = (0, 80, 255) if ("Falls" in text and self.total_falls > 0) else (190, 190, 190)
            cv2.putText(frame, text, (xp, my + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1)

        cv2.putText(frame, f"Frames: {self.total_frames}  |  Stand conf: {stand_conf * 100:.0f}%",
                    (30, my + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (110, 110, 110), 1)

        # ── Bounding box ──────────────────────────────────────────────────────
        if self.use_custom and bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            bc = (0, 80, 255) if is_fall else (0, 220, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bc, 3 if is_fall else 2)
            label = f"{CLASS_NAMES[cls_id] if cls_id is not None and cls_id < 3 else 'person'}: {cls_conf:.0%}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 6, y1 - 2), bc, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 1)

        # ── Pose skeleton ─────────────────────────────────────────────────────
        elif keypoints is not None and len(keypoints) > 0:
            kps    = keypoints[0]
            pairs  = [(5,6),(5,11),(6,12),(11,12),(5,7),(7,9),(6,8),(8,10),
                      (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
            pts    = []
            sk_col = (0, 80, 255) if is_fall else (0, 220, 0)

            for kp in kps:
                if kp[2] > 0.25:
                    px = int(kp[0] * w / 640)
                    py = int(kp[1] * h / 480)
                    pts.append((px, py))
                    cv2.circle(frame, (px, py), 4, sk_col, -1)
                else:
                    pts.append(None)

            for a, b in pairs:
                if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                    cv2.line(frame, pts[a], pts[b], sk_col, 2)

        # ── Low-light warning ─────────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if float(np.mean(gray)) < 70:
            cv2.putText(frame, "LOW LIGHT", (w - 130, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)

        return frame

    # ── FPS ───────────────────────────────────────────────────────────────────
    def _calc_fps(self) -> int:
        self.fps_count += 1
        if time.time() - self.fps_start >= 1.0:
            self.fps       = self.fps_count
            self.fps_count = 0
            self.fps_start = time.time()
        return self.fps


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    PATIENT_ID = 1

    print("\n" + "=" * 78)
    print("  🏥  GUARDIAN NET — ENHANCED FALL DETECTION PRO")
    print("=" * 78)
    print("  ✓  Multi-feature fusion (aspect ratio + angle + velocity + ground proximity)")
    print("  ✓  Temporal smoothing to eliminate false positives")
    print("  ✓  Adaptive state machine with hysteresis")
    print("  ✓  Cross-platform emergency alarm")
    print("  ✓  Custom YOLO model support (human_fall_detection dataset)")
    print("=" * 78 + "\n")

    detector = FallDetectorPro(patient_id=PATIENT_ID)

    # ── Open camera ───────────────────────────────────────────────────────────
    cap = None
    backend = cv2.CAP_DSHOW if platform.system() == "Windows" else 0

    for idx in range(3):
        try:
            c = cv2.VideoCapture(idx, backend) if platform.system() == "Windows" \
                else cv2.VideoCapture(idx)
            if c.isOpened():
                ret, frm = c.read()
                if ret and frm is not None:
                    cap = c
                    print(f"✅ Camera index {idx} opened.")
                    break
            c.release()
        except Exception:
            pass

    if cap is None:
        print("❌ No camera found. Check connection and try again.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\n  🎥 LIVE — Press Q to quit | R to reset counter | S for status\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            fall_c, stand_c, kps, bbx, cid, ccf = detector.process_frame(frame)
            frame = detector.draw_ui(frame, fall_c, stand_c, kps, bbx, cid, ccf)

            cv2.imshow("Guardian Net — Fall Detection Pro", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                detector.total_falls = 0
                print("📊 Fall counter reset.")
            elif key == ord("s"):
                dr = (detector.detection_frames / max(1, detector.total_frames)) * 100
                print(f"\n📊 Status | Model: {detector.model_label} | "
                      f"Frames: {detector.total_frames} | Det rate: {dr:.1f}% | "
                      f"Falls: {detector.total_falls} | "
                      f"Fall conf: {fall_c:.2%}\n")

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 78)
        print("  📊  SESSION REPORT")
        print("=" * 78)
        print(f"  Falls detected  : {detector.total_falls}")
        print(f"  Alerts sent     : {detector.alert_sender.alert_count}")
        print(f"  Model used      : {detector.model_label}")
        print(f"  Total frames    : {detector.total_frames}")
        print("=" * 78)
        print("  👋  Shutdown complete. Stay safe!\n")


if __name__ == "__main__":
    main()