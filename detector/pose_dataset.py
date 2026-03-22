"""
Download Pose-based Fall Detection Dataset
"""

import os
import urllib.request
import zipfile
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    print(f"📥 Downloading {filename}...")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for data in response.iter_content(chunk_size=1024):
                f.write(data)
                pbar.update(len(data))
    
    print(f"✅ Downloaded {filename}")

def download_multicam_fall():
    """Download Multicam Fall Dataset"""
    print("\n" + "="*70)
    print("📥 DOWNLOADING MULTICAM FALL DATASET")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset', 'pose_fall_dataset')
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Multicam Fall Dataset URLs (contains pose annotations)
    # Note: You may need to request access from the authors
    urls = [
        'http://multicam-fall-dataset.com/dataset/fall_videos.zip',
        # Add more URLs as needed
    ]
    
    for url in urls:
        filename = os.path.join(dataset_dir, url.split('/')[-1])
        try:
            download_file(url, filename)
            
            # Extract if zip
            if filename.endswith('.zip'):
                print(f"📦 Extracting {filename}...")
                with zipfile.ZipFile(filename, 'r') as zip_ref:
                    zip_ref.extractall(dataset_dir)
                print("✅ Extraction complete")
                
        except Exception as e:
            print(f"⚠️ Failed to download from {url}: {e}")
    
    return dataset_dir

def create_custom_pose_dataset():
    """Create a simple pose dataset from videos"""
    print("\n" + "="*70)
    print("🎥 CREATE CUSTOM POSE DATASET")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset', 'pose_fall_dataset')
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, 'valid'), exist_ok=True)
    
    print(f"\n📁 Created dataset structure at: {dataset_dir}")
    print("\n📝 To create a pose dataset, you need:")
    print("   1. Videos of people falling and standing")
    print("   2. Annotate keypoints (shoulders, hips, knees, ankles)")
    print("\n💡 Alternative: Use existing pose datasets:")
    print("   - COCO-pose dataset (has person keypoints)")
    print("   - MPII Human Pose dataset")
    print("   - AI Challenger dataset")
    
    return dataset_dir

def setup_coco_pose():
    """Download COCO-pose dataset subset"""
    print("\n" + "="*70)
    print("📥 SETTING UP COCO-POSE DATASET")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_dir = os.path.join(project_root, 'dataset', 'coco_pose')
    
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\n📋 To use COCO-pose dataset:")
    print("\n1. Download annotations:")
    print("   wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
    print("\n2. Download person keypoints annotations:")
    print("   wget http://images.cocodataset.org/annotations/person_keypoints_trainval2017.zip")
    print("\n3. Download some images:")
    print("   wget http://images.cocodataset.org/zips/train2017.zip")
    print("\n4. Extract and convert to YOLO-pose format")
    
    print(f"\n📁 Dataset folder: {dataset_dir}")
    
    return dataset_dir

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏋️ POSE FALL DETECTION DATASET SETUP")
    print("="*70)
    print("\nOptions:")
    print("1. Setup COCO-pose dataset (recommended)")
    print("2. Create custom pose dataset")
    print("3. Try Multicam Fall dataset")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        setup_coco_pose()
    elif choice == '2':
        create_custom_pose_dataset()
    elif choice == '3':
        download_multicam_fall()
    else:
        print("\n👋 Goodbye!")