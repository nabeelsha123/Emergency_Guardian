"""
GUARDIAN NET — MODEL INSPECTOR
Run this FIRST to identify your custom model type and classes.
Share the output so the correct fall detection logic can be written.

Usage:
    python inspect_model.py
"""

import sys, os, numpy as np

MODEL_PATH = r"C:\Users\nabee\OneDrive\Desktop\Guardian Emergency\runs\train\fall_custom_scratch\weights\best.pt"

def inspect():
    print("\n" + "="*60)
    print("🔍 GUARDIAN NET — MODEL INSPECTOR")
    print("="*60)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed.  pip install ultralytics")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found:\n   {MODEL_PATH}")
        sys.exit(1)

    print(f"\n📦 Loading: {MODEL_PATH}\n")
    model = YOLO(MODEL_PATH)

    task  = getattr(model, 'task', 'unknown')
    names = model.names
    nc    = len(names)

    print(f"🧠 Task       : {task}")
    print(f"🏷️  Classes    : {names}")
    print(f"🔢 Num classes: {nc}")

    try:
        imgsz = model.overrides.get('imgsz', '?')
        print(f"📐 Input size : {imgsz}")
    except Exception:
        pass

    # Blank frame inference
    blank = np.zeros((640,640,3), dtype=np.uint8)
    print("\n🎯 Blank-frame inference...")
    r = model(blank, verbose=False)[0]

    has_boxes = r.boxes is not None and len(r.boxes) > 0
    has_kpts  = hasattr(r,'keypoints') and r.keypoints is not None
    has_masks = hasattr(r,'masks')     and r.masks     is not None

    print(f"   boxes     : {'yes  ← DETECTION / BBox model' if r.boxes is not None else 'no'}")
    print(f"   keypoints : {'yes  ← POSE model'             if has_kpts             else 'none'}")
    print(f"   masks     : {'yes  ← SEGMENTATION model'     if has_masks            else 'none'}")

    # Webcam test
    print("\n📷 Webcam frame test (index 0)...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            r2 = model(frame, verbose=False)[0]
            boxes = r2.boxes
            if boxes is not None and len(boxes) > 0:
                print(f"   ✅ {len(boxes)} detection(s):")
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = names.get(cls_id, f"class_{cls_id}")
                    print(f"      [{i}] class={cls_id} ({label})  conf={conf:.2f}")
            else:
                print("   ℹ️  No detections (stand in front of camera and re-run)")
        else:
            print("   ⚠️  Webcam not readable")
    except Exception as e:
        print(f"   ⚠️  Webcam test error: {e}")

    print("\n" + "="*60)
    print("✅ Copy and share this entire output.")
    print("="*60 + "\n")

if __name__ == "__main__":
    inspect()