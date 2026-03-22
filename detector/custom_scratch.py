"""
Train Fall Detection Model FROM SCRATCH (No Pretrained Model)
This is what your college wants - completely custom training
"""

import os
import yaml
from ultralytics import YOLO
import torch

def train_from_scratch():
    print("\n" + "="*70)
    print("🎯 TRAINING CUSTOM MODEL FROM SCRATCH")
    print("   No pretrained models - completely original training")
    print("="*70)
    
    # Get paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Dataset path
    dataset_path = os.path.join(project_root, 'dataset', 'human-fall-detection', 'data.yaml')
    
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found at: {dataset_path}")
        return False
    
    # Load dataset info
    with open(dataset_path, 'r') as f:
        data = yaml.safe_load(f)
    
    print(f"\n📊 Dataset Information:")
    print(f"   Classes: {data.get('names', ['fallen', 'sitting', 'standing'])}")
    print(f"   Number of classes: {data.get('nc', 3)}")
    
    # Count training images
    dataset_dir = os.path.dirname(dataset_path)
    train_images = os.path.join(dataset_dir, 'train', 'images')
    if os.path.exists(train_images):
        img_count = len([f for f in os.listdir(train_images) if f.endswith(('.jpg', '.png', '.jpeg'))])
        print(f"   Training images: {img_count}")
    
    print("\n" + "-"*70)
    print("⚠️ IMPORTANT: Training from scratch (no pretrained weights)")
    print("   This will take longer but is what your college requires")
    print("-"*70)
    
    # Training options
    print("\nTraining Options:")
    print("1. Quick training (20 epochs) - ~30 minutes")
    print("2. Standard training (50 epochs) - ~1-2 hours")
    print("3. Extensive training (100 epochs) - ~3-4 hours")
    
    choice = input("\nSelect training mode (1-3): ").strip()
    
    if choice == '1':
        epochs = 20
    elif choice == '2':
        epochs = 50
    elif choice == '3':
        epochs = 100
    else:
        epochs = 50
    
    print(f"\n🎯 Training for {epochs} epochs from scratch")
    
    # Check for GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    if device == '0':
        print(f"✅ GPU detected - training will be faster")
        batch_size = 16
    else:
        print(f"⚠️ No GPU - training on CPU")
        batch_size = 8
    
    # ==================== KEY: NO PRETRAINED MODEL ====================
    # Create a brand new YOLO model (no pretrained weights)
    print("\n🔄 Creating new model from scratch...")
    print("   No pretrained weights - training from random initialization")
    
    # Method 1: Use a YAML config (completely fresh)
    model = YOLO('yolov8n.yaml')  # This creates a fresh model from config
    
    # Alternative method if the above doesn't work:
    # model = YOLO('yolov8n.pt')  # DON'T use this - it loads pretrained!
    
    print("   ✅ Fresh model created")
    print("   Model architecture: YOLOv8n (from scratch)")
    
    # Start training
    print(f"\n🚀 Starting training from scratch...")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print("\n   This will take time. Press Ctrl+C to stop\n")
    
    try:
        results = model.train(
            data=dataset_path,
            epochs=epochs,
            imgsz=640,
            batch=batch_size,
            device=device,
            workers=4,
            project=os.path.join(project_root, 'runs', 'train'),
            name='fall_custom_scratch',
            exist_ok=True,
            pretrained=False,  # IMPORTANT: No pretrained weights!
            optimizer='SGD',
            lr0=0.01,
            patience=10,
            verbose=True
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        # Model location
        model_path = os.path.join(project_root, 'runs', 'train', 'fall_custom_scratch', 'weights', 'best.pt')
        print(f"\n📁 Model saved to: {model_path}")
        print(f"\n📊 This model was trained from scratch (no pretrained weights)")
        
        # Evaluate
        print("\n📊 Evaluating model...")
        metrics = model.val()
        
        print("\n📈 Results:")
        print(f"   mAP50: {metrics.box.map50:.3f}")
        print(f"   mAP50-95: {metrics.box.map:.3f}")
        
        if metrics.box.map50 > 0.7:
            print("   🎉 Excellent! Model is very accurate")
        elif metrics.box.map50 > 0.5:
            print("   👍 Good model. Ready for demo")
        else:
            print("   ⚠️ Accuracy is low. Consider training more epochs")
        
        print("\n✅ Model is ready for your demo!")
        print("\nTo use this model, run:")
        print("   python detector/guardian_fall_custom.py")
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted")
        return False
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*70)
    print("🏋️ CUSTOM MODEL TRAINER (FROM SCRATCH)")
    print("="*70)
    print("\nThis script trains a completely custom model")
    print("✅ No pretrained weights used")
    print("✅ Everything trained on YOUR dataset")
    print("✅ Perfect for college requirements\n")
    
    train_from_scratch()

if __name__ == "__main__":
    main()