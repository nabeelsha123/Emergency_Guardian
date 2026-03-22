#!/usr/bin/env python
"""
Guardian Net - Enhanced Fall Detection + Voice Detection (Fixed)
════════════════════════════════════════════════════════════════
Voice fixes applied:
  1. adjust_for_ambient_noise called EVERY loop iteration → silences mic for 0.3s
     each cycle, blocking actual listening. Now called only ONCE at startup.
  2. timeout=1 means only 1 second wait for speech to START — too short when
     background noise fills the buffer. Raised to timeout=3.
  3. phrase_time_limit=3 cuts off longer phrases like "please help me I fell".
     Raised to 5 seconds.
  4. Silent bare except: swallows ALL errors including network, so you never
     see why transcription fails. Now logs every real error to terminal.
  5. winsound.Beep blocks the voice thread for 500ms, causing missed audio.
     Moved to a daemon thread.
  6. Microphone index hardcoded to default — added mic diagnostics at startup
     so you can see exactly which mic is being used.
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
#  CONSTANTS  — balanced to catch real falls, ignore sitting/bending
# ══════════════════════════════════════════════════════════════════════════════
FRAME_W, FRAME_H   = 640, 480

# Original values — 0.55 confidence, 5 frames
FALL_CONF_THRESH   = 0.55

# 5 frames (~0.2s) — original setting
REQUIRED_FALL_FRM  = 5

REQUIRED_STAND_FRM = 10
ALERT_COOLDOWN     = 30

AR_FALL_MAX        = 0.90
AR_FALL_ZONE_MAX   = 1.20
AR_STAND_MIN       = 1.80
ANGLE_FALL_MIN     = 50
ANGLE_STAND_MAX    = 22

CLASS_NAMES        = ["fallen", "sitting", "standing"]


# ══════════════════════════════════════════════════════════════════════════════
#  FALL DETECTOR  (unchanged from previous version)
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
        self.prev_bbox     = None
        self.prev_time     = None
        self.running       = True
        self.frame_queue   = queue.Queue(maxsize=2)
        print(f"   ✅ Fall Detection Ready [{self.model_label}]")

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

    def _fall_conf_bbox(self, bbox, cls_id, cls_conf):
        x1,y1,x2,y2 = bbox
        w_b=max(1,x2-x1); h_b=max(1,y2-y1)
        scores=[]

        # ── 1. Aspect ratio score ──────────────────────────────────────────
        ar=h_b/w_b
        if ar<AR_FALL_MAX:          ar_s=1.0
        elif ar<AR_FALL_ZONE_MAX:   ar_s=1.0-(ar-AR_FALL_MAX)/(AR_FALL_ZONE_MAX-AR_FALL_MAX)
        else:                       ar_s=0.0
        scores.append(ar_s*0.35)

        # ── 2. Ground proximity ────────────────────────────────────────────
        cy_norm=((y1+y2)/2)/FRAME_H
        if cy_norm>0.60: scores.append(min(1.0,(cy_norm-0.50)*2.0)*0.20)

        # ── 3. Downward velocity (must be significant to count) ────────────
        if self.prev_bbox is not None and self.prev_time is not None:
            dt=max(0.02,time.time()-self.prev_time)
            ph=max(1,self.prev_bbox[3]-self.prev_bbox[1])
            h_loss=(ph-h_b)/ph; down=(y2-self.prev_bbox[3])/ph; vel=abs(down)/dt
            self.velocity_hist.append(vel); avg_vel=float(np.mean(self.velocity_hist))
            # Raised thresholds: needs bigger height loss AND faster movement
            if h_loss>0.20 and avg_vel>0.18:
                scores.append(min(1.0,h_loss*1.5+avg_vel)*0.25)

        # ── 4. Model class confidence ──────────────────────────────────────
        if cls_id==0 and cls_conf>0.45:
            # "fallen" class detected with good confidence → boost score
            scores.append(cls_conf*0.20)
        elif cls_id==1:
            # "sitting" class detected → CANCEL the fall score entirely
            # This is the key fix: if the model says "sitting", it's not a fall
            return 0.0
        elif cls_id==2 and cls_conf>0.40:
            # "standing" class → also cancel
            return 0.0

        raw=min(1.0,sum(scores)); self.fall_hist.append(raw)
        return float(np.mean(self.fall_hist)) if len(self.fall_hist)>=2 else raw

    def _stand_conf_bbox(self, bbox, cls_id, cls_conf):
        x1,y1,x2,y2=bbox; w_b=max(1,x2-x1); h_b=max(1,y2-y1)
        scores=[]; ar=h_b/w_b
        if ar>AR_STAND_MIN: scores.append(min(1.0,(ar-AR_STAND_MIN)/0.6+0.4)*0.70)
        if cls_id==2 and cls_conf>0.35: scores.append(cls_conf*0.30)
        raw=float(np.mean(scores)) if scores else 0.0; self.stand_hist.append(raw)
        return float(np.mean(self.stand_hist)) if len(self.stand_hist)>=2 else raw

    def _fall_conf_pose(self, keypoints, frame_shape):
        if keypoints is None or len(keypoints)==0: return 0.0
        kps=keypoints[0]; fh,fw=frame_shape[:2]; scores=[]
        if len(kps)>=13:
            ls,rs,lh,rh=kps[5],kps[6],kps[11],kps[12]
            if all(k[2]>0.25 for k in [ls,rs,lh,rh]):
                sh_c=(ls[:2]+rs[:2])/2; hp_c=(lh[:2]+rh[:2])/2
                dx,dy=hp_c[0]-sh_c[0],hp_c[1]-sh_c[1]
                angle=math.degrees(math.atan2(abs(dx),max(1e-4,abs(dy))))
                self.angle_hist.append(angle); avg_a=float(np.mean(self.angle_hist))
                if avg_a>ANGLE_FALL_MIN: scores.append(min(1.0,(avg_a-30)/55)*0.40)
        valid=[k for k in kps if k[2]>0.25]
        if len(valid)>=4:
            ys=[k[1] for k in valid]; xs=[k[0] for k in valid]
            kh=max(ys)-min(ys)+1e-4; kw=max(xs)-min(xs)+1e-4; ar=kh/kw
            if ar<1.20:              scores.append((1.0-ar/1.20)*0.30)
            elif ar<AR_FALL_ZONE_MAX: scores.append((1.0-(ar-1.20)/0.45)*0.15)
        if len(kps)>=17:
            ankles=[kps[i] for i in [15,16] if kps[i][2]>0.25]
            heads =[kps[i] for i in [3,4]   if kps[i][2]>0.25]
            if ankles and heads:
                ank_y=max(k[1] for k in ankles); hd_y=min(k[1] for k in heads)
                if ank_y>hd_y: scores.append(min(1.0,(ank_y/fh)*1.25)*0.30)
        raw=min(1.0,sum(scores)); self.fall_hist.append(raw)
        return float(np.mean(self.fall_hist)) if len(self.fall_hist)>=2 else raw

    def _stand_conf_pose(self, keypoints):
        if keypoints is None or len(keypoints)==0: return 0.0
        kps=keypoints[0]; scores=[]
        if len(kps)>=13:
            ls,rs,lh,rh=kps[5],kps[6],kps[11],kps[12]
            if all(k[2]>0.25 for k in [ls,rs,lh,rh]):
                sh_c=(ls[:2]+rs[:2])/2; hp_c=(lh[:2]+rh[:2])/2
                dx,dy=hp_c[0]-sh_c[0],hp_c[1]-sh_c[1]
                angle=math.degrees(math.atan2(abs(dx),max(1e-4,abs(dy))))
                if angle<ANGLE_STAND_MAX:  scores.append(1.0*0.70)
                elif angle<40:             scores.append((1.0-(angle-ANGLE_STAND_MAX)/15)*0.70)
        valid=[k for k in kps if k[2]>0.25]
        if len(valid)>=4:
            ys=[k[1] for k in valid]; scores.append(min(1.0,(max(ys)-min(ys))/320)*0.30)
        raw=float(np.mean(scores)) if scores else 0.0; self.stand_hist.append(raw)
        return float(np.mean(self.stand_hist)) if len(self.stand_hist)>=2 else raw

    def _update_state(self, fall_conf, stand_conf):
        now=time.time()
        if self.state=="MONITORING":
            if fall_conf>FALL_CONF_THRESH: self.consecutive_fall_frms+=1
            else: self.consecutive_fall_frms=max(0,self.consecutive_fall_frms-1)
            if self.consecutive_fall_frms>=REQUIRED_FALL_FRM:
                self.state="FALL_DETECTED"; self.fall_start_time=now
                self.total_falls+=1; self.consecutive_stand_frms=0
                if now-self.last_alert_time>ALERT_COOLDOWN:
                    self.last_alert_time=now
                    msg=f"🚨 Fall detected with {fall_conf:.1%} confidence!"
                    alert_queue.put(("fall",msg,float(fall_conf)))
                    play_alarm_sound("fall")
                    print(f"\n🔴 FALL DETECTED! Conf={fall_conf:.2%} | Total={self.total_falls}")
        elif self.state=="FALL_DETECTED":
            if stand_conf>0.60: self.consecutive_stand_frms+=1
            else: self.consecutive_stand_frms=max(0,self.consecutive_stand_frms-1)
            if self.consecutive_stand_frms>=REQUIRED_STAND_FRM:
                self._clear_fall("   ✅ Person stood up — resuming monitoring.")
            elif now-self.fall_start_time>30:
                self._clear_fall("   🔄 Fall timeout — resuming monitoring.")

    def _clear_fall(self, msg):
        self.state="MONITORING"; self.consecutive_fall_frms=0
        self.consecutive_stand_frms=0; self.fall_hist.clear(); print(msg)

    def process_frame(self, frame):
        proc=cv2.resize(frame,(FRAME_W,FRAME_H))
        results=self.model(proc,verbose=False,conf=0.30,imgsz=640)
        fall_conf=stand_conf=0.0; keypoints=bbox=None; cls_id=None; cls_conf=0.0
        if self.use_custom:
            if results and results[0].boxes is not None:
                best=0.0
                for box in results[0].boxes:
                    c=float(box.conf[0]); ci=int(box.cls[0])
                    if c>best:
                        best=c; x1,y1,x2,y2=box.xyxy[0].cpu().numpy()
                        bbox=(x1,y1,x2,y2); cls_id=ci; cls_conf=c
                if bbox:
                    fall_conf=self._fall_conf_bbox(bbox,cls_id,cls_conf)
                    stand_conf=self._stand_conf_bbox(bbox,cls_id,cls_conf)
        else:
            if results and results[0].keypoints is not None:
                kps=results[0].keypoints.data.cpu().numpy()
                if len(kps)>0:
                    keypoints=kps
                    fall_conf=self._fall_conf_pose(keypoints,proc.shape)
                    stand_conf=self._stand_conf_pose(keypoints)
        if bbox is not None: self.prev_bbox=bbox; self.prev_time=time.time()
        self._update_state(fall_conf,stand_conf)
        self.shared_state['fall']={
            'state':self.state,'confidence':fall_conf,
            'total':self.total_falls,'model':self.model_label}
        return fall_conf,stand_conf,keypoints,bbox,cls_id,cls_conf

    def fall_detection_loop(self):
        cap=None
        backend=cv2.CAP_DSHOW if platform.system()=="Windows" else 0
        for idx in range(3):
            try:
                c=(cv2.VideoCapture(idx,backend) if platform.system()=="Windows"
                   else cv2.VideoCapture(idx))
                if c.isOpened():
                    ret,frm=c.read()
                    if ret and frm is not None:
                        cap=c; print(f"   ✅ Fall camera at index {idx}"); break
                c.release()
            except Exception: pass
        if cap is None:
            print("   ❌ Cannot open camera"); self.running=False; return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,FRAME_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,FRAME_H)
        cap.set(cv2.CAP_PROP_FPS,30)
        err=0
        while self.running:
            try:
                ret,frame=cap.read()
                if not ret or frame is None:
                    err+=1
                    if err>10: print("   ⚠️ Camera lost"); break
                    time.sleep(0.1); continue
                err=0; self.process_frame(frame)
                if not self.frame_queue.full(): self.frame_queue.put(frame.copy())
            except Exception as e:
                print(f"   ⚠️ Fall error: {e}"); time.sleep(0.5)
        cap.release(); print("   👋 Fall detection stopped")

    def stop(self):
        self.running=False


# ══════════════════════════════════════════════════════════════════════════════
#  VOICE DETECTOR  — ALL 6 BUGS FIXED (see comments inline)
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

        # ══════════════════════════════════════════════════════════════════
        #  SET YOUR MIC INDEX HERE
        #  Run this in PowerShell to find your device number:
        #    python -c "import speech_recognition as sr; [print(f'[{i}] {m}') for i,m in enumerate(sr.Microphone.list_microphone_names())]"
        #
        #  Common values:
        #    MIC_INDEX = None   → Windows default mic (works without headset)
        #    MIC_INDEX = 2      → Airdopes 181 / Headset (Bluetooth)
        #    MIC_INDEX = 26     → Realtek built-in mic
        # ══════════════════════════════════════════════════════════════════
        MIC_INDEX = 2       # ← change this number for a different headset

        # Validate index exists — fall back to default if device is missing
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

        # Calibrate once at startup
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

    # ── Language helpers (unchanged) ──────────────────────────────────────────
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
        """
        Try primary language first, then fall back to others.
        FIX 4: Replaced bare except with specific catches so errors are visible.
        """
        # Primary language attempt
        try:
            text = self.recognizer.recognize_google(
                audio, language=self.current_language)
            return text, True
        except sr.UnknownValueError:
            pass   # speech heard but unintelligible — normal, not an error
        except sr.RequestError as e:
            print(f"   ⚠️  Google Speech API error: {e}")
            print("         Check internet connection.")
            return None, False
        except Exception as e:
            print(f"   ⚠️  Transcription error: {e}")
            return None, False

        # Fallback: try other languages
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
            # Recreate mic object each outer loop — recovers from device errors
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
                # Device disconnected mid-session — log once and retry
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
    shared_state.setdefault('fall',  {'state': 'MONITORING', 'confidence': 0,
                                      'total': 0, 'model': ''})

    fps_start = time.time(); fps_count = 0; fps = 0

    while fall_detector.running:
        try:
            frame = fall_detector.frame_queue.get(timeout=1)
            disp  = frame.copy()
            h, w  = disp.shape[:2]

            fps_count += 1
            if time.time() - fps_start >= 1.0:
                fps = fps_count; fps_count = 0; fps_start = time.time()

            fall_state  = shared_state.get('fall',  {})
            voice_state = shared_state.get('voice', {})
            is_fall     = fall_state.get('state') == "FALL_DETECTED"
            fall_conf   = float(fall_state.get('confidence', 0))

            # Top bar
            ov = disp.copy()
            cv2.rectangle(ov, (0,0),(w,90),(10,10,10),-1)
            cv2.addWeighted(ov, 0.75, disp, 0.25, 0, disp)
            cv2.putText(disp,"GUARDIAN NET",(w//2-105,30),
                        cv2.FONT_HERSHEY_DUPLEX,0.75,(80,220,80),2)
            cv2.putText(disp,"FALL & VOICE DETECTION",(w//2-130,60),
                        cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
            cv2.putText(disp,f"{fps} FPS",(w-85,28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(80,220,80),1)

            # Model badge
            mdl   = fall_state.get('model','')
            is_cu = mdl not in ('','POSE')
            cv2.putText(disp,f"[{mdl}]",(10,28),
                        cv2.FONT_HERSHEY_SIMPLEX,0.42,
                        (60,220,60) if is_cu else (0,200,220),1)
            cv2.putText(disp,"CUSTOM" if is_cu else "POSE",(10,46),
                        cv2.FONT_HERSHEY_SIMPLEX,0.32,
                        (60,220,60) if is_cu else (0,200,220),1)

            # Fall status card
            cy,ch = 98,65; pulse=int(time.time()*4)%2
            if is_fall:
                bg=(20,0,0) if pulse else (40,0,0); bdr=(0,60,255)
                txt="FALL DETECTED!"; sub="EMERGENCY ALERT SENT"; tc=(0,80,255)
            else:
                bg=(0,28,0); bdr=(0,180,0)
                txt="MONITORING"; sub="No fall detected"; tc=(80,240,80)
            cv2.rectangle(disp,(30,cy),(w-30,cy+ch),bg,-1)
            cv2.rectangle(disp,(30,cy),(w-30,cy+ch),bdr,2)
            cv2.putText(disp,txt,(w//2-110,cy+28),cv2.FONT_HERSHEY_DUPLEX,0.80,tc,2)
            cv2.putText(disp,sub,(w//2-130,cy+52),cv2.FONT_HERSHEY_SIMPLEX,0.46,tc,1)

            # Confidence gauge
            gy=cy+ch+14; gw=w-70
            cv2.rectangle(disp,(35,gy),(35+gw,gy+20),(35,35,35),-1)
            cv2.rectangle(disp,(35,gy),(35+gw,gy+20),(70,70,70),1)
            fill=int(gw*min(1.0,fall_conf))
            if fill>0:
                cv2.rectangle(disp,(35,gy),(35+fill,gy+20),
                              (0,70,255) if is_fall else (0,200,0),-1)
            tx=35+int(gw*FALL_CONF_THRESH)
            cv2.line(disp,(tx,gy-4),(tx,gy+24),(255,165,0),2)
            cv2.putText(disp,f"FALL CONF: {fall_conf*100:.0f}%  [thresh {FALL_CONF_THRESH*100:.0f}%]",
                        (35,gy-6),cv2.FONT_HERSHEY_SIMPLEX,0.43,(210,210,210),1)

            # Voice status row — shows live transcription status
            vy=gy+20+14
            v_status=voice_state.get('status','Listening')
            if "EMERGENCY" in v_status or "emergency" in v_status.lower():
                vc=(0,60,255)
            elif "Heard" in v_status:
                vc=(0,220,255)
            elif "Processing" in v_status:
                vc=(0,255,180)
            elif "Error" in v_status:
                vc=(0,100,255)
            else:
                vc=(0,160,200)
            cv2.putText(disp,f"MIC: {v_status}",(35,vy+14),
                        cv2.FONT_HERSHEY_SIMPLEX,0.50,vc,1)

            # Stats strip
            sy=vy+28
            cv2.rectangle(disp,(20,sy),(w-20,sy+48),(12,12,12),-1)
            cols=[
                (f"Patient: {fall_detector.alert_sender.patient_id}",30),
                (f"Falls: {fall_state.get('total',0)}",              30+w//4),
                (f"Voice alerts: {voice_state.get('total',0)}",      30+w//2),
                (f"Sent: {fall_detector.alert_sender.alert_count}",  30+3*w//4),
            ]
            for t2,xp in cols:
                c2=((0,80,255) if "Falls" in t2 and fall_state.get('total',0)>0 else
                    (0,200,255) if "Voice" in t2 and voice_state.get('total',0)>0 else
                    (185,185,185))
                cv2.putText(disp,t2,(xp,sy+22),cv2.FONT_HERSHEY_SIMPLEX,0.43,c2,1)

            cv2.putText(disp,"Q=quit",(w-65,h-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.40,(100,100,100),1)

            cv2.imshow("Guardian Net - Fall & Voice Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                fall_detector.running=False; break

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Display error: {e}"); break

    cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Accept patient_id from command line so Node.js can pass it ──────────
    # Usage:  python guardian_all.py --patient_id 3
    # Falls back to 1 if not supplied (manual run).
    import argparse
    parser = argparse.ArgumentParser(description="Guardian Net Detector")
    parser.add_argument("--patient_id", type=int, default=1,
                        help="Patient ID to monitor (default: 1)")
    args, _ = parser.parse_known_args()
    patient_id = args.patient_id

    print("\n" + "="*70)
    print("🚀 GUARDIAN NET — ENHANCED FALL + VOICE DETECTION  (v3 Fixed)")
    print("="*70)
    print(f"\n📱 Patient ID: {patient_id}")
    print("="*70)

    alert_sender = GuardianAlertSender(patient_id=patient_id)
    if hasattr(alert_sender,'test_connection'):
        if alert_sender.test_connection():
            print("✅ Connected to Guardian Net server")
        else:
            print("⚠️  Cannot connect — alerts logged only")

    shared_state = {
        'voice': {'status':'Starting...','total':0},
        'fall':  {'state':'Starting...','confidence':0,'total':0,'model':''},
    }

    print("\n🔧 Initializing detectors...")
    fall_detector  = UnifiedFallDetector(alert_sender, shared_state)
    voice_detector = UnifiedVoiceDetector(alert_sender, shared_state)

    print("\n" + "="*70)
    print("✅ ALL DETECTORS READY — Starting threads...")
    print("="*70)
    print("📹 Fall  : Camera")
    print("🎤 Voice : Microphone  (EN / Malayalam / Hindi)")
    print("\nPress Q in the video window to quit")
    print("="*70 + "\n")

    threads = [
        threading.Thread(target=fall_detector.fall_detection_loop,   daemon=True, name="Fall"),
        threading.Thread(target=voice_detector.voice_detection_loop,  daemon=True, name="Voice"),
        threading.Thread(target=alert_handler, args=(alert_sender,),  daemon=True, name="Alert"),
        threading.Thread(target=display_thread,
                         args=(fall_detector, shared_state),          daemon=True, name="Display"),
    ]
    for t in threads: t.start()

    try:
        while fall_detector.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping...")

    fall_detector.stop()
    voice_detector.stop()

    print("\n📊 Final Summary")
    print("="*50)
    print(f"   Falls detected   : {fall_detector.total_falls}")
    print(f"   Voice emergencies: {voice_detector.emergency_count}")
    print(f"   Alerts sent      : {alert_sender.alert_count}")
    print("="*50)
    print("\n👋 Goodbye!\n")


if __name__ == "__main__":
    main()