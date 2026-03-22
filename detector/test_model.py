"""
Test Custom Model on Images
This will help verify if your model is working correctly
"""

import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

def test_model_on_images():
    print("\n" + "="*70)
    print("🔍 TESTING CUSTOM MODEL")
    print("="*70)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Model path
    model_path = os.path.join(project_root, 'runs', 'train', 'fall_detection', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found at: {model_path}")
        return
    
    print(f"\n✅ Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Test on validation images
    valid_images = os.path.join(project_root, 'dataset', 'human-fall-detection', 'valid', 'images')
    
    if not os.path.exists(valid_images):
        print(f"\n❌ Test images not found at: {valid_images}")
        print("\nTrying train images instead...")
        valid_images = os.path.join(project_root, 'dataset', 'human-fall-detection', 'train', 'images')
    
    if not os.path.exists(valid_images):
        print("\n❌ No test images found!")
        return
    
    # Get some test images
    images = [f for f in os.listdir(valid_images) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not images:
        print("❌ No images found")
        return
    
    print(f"\n📸 Found {len(images)} test images")
    print("\nTesting on 5 sample images...\n")
    
    # Test on first 5 images
    for img_file in images[:5]:
        img_path = os.path.join(valid_images, img_file)
        print(f"Testing: {img_file}")
        
        # Run inference
        results = model(img_path)
        
        # Show results
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            print(f"   ✅ Found {len(boxes)} detections:")
            
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = ['fallen', 'sitting', 'standing'][cls]
                print(f"      - {class_name}: {conf:.2f}")
        else:
            print(f"   ⚠️ No detections found")
        
        # Display the image with detections
        results[0].show()
        cv2.waitKey(1000)
    
    print("\n✅ Test complete!")
    print("\nIf you see detections, your model is working.")
    print("If not, the model needs more training.")

def test_with_camera():
    """Test model with camera and show live detections"""
    print("\n" + "="*70)
    print("🎥 TESTING MODEL WITH CAMERA")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    model_path = os.path.join(project_root, 'runs', 'train', 'fall_detection', 'weights', 'best.pt')
    
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found")
        return
    
    print(f"\n✅ Loading model...")
    model = YOLO(model_path)
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return
    
    print("\n🎥 Camera started. Press 'q' to quit")
    print("📊 Showing live detections...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        
        # Draw results
        annotated = results[0].plot()
        
        # Add info text
        cv2.putText(annotated, "Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Count detections
        if results[0].boxes is not None:
            detections = len(results[0].boxes)
            cv2.putText(annotated, f"Detections: {detections}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Model Test - Press 'q' to quit", annotated)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Camera test complete!")

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🔍 CUSTOM MODEL TESTER")
    print("="*70)
    print("\nOptions:")
    print("1. Test on dataset images")
    print("2. Test with camera (live)")
    print("3. Exit")
    
    choice = input("\nChoose (1-3): ").strip()
    
    if choice == '1':
        test_model_on_images()
    elif choice == '2':
        test_with_camera()
    else:
        print("👋 Goodbye!")