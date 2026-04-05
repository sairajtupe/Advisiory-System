import os
import shutil
import random

# ==========================================
# ⚠️ STEP 1: SET YOUR FOLDER PATHS HERE
# ==========================================

# 1. Where did you extract the Kaggle download? 
SOURCE_FOLDER = r"F:\crop dataset\Sugarcane_leafs\Mosaic"
# 2. Where is your project's dataset folder?
TARGET_FOLDER = r"F:\project\smart_crop_advisor\dataset"

# 3. EXACT TARGET FOLDER NAME (e.g., "cotton_disease", "tomato_healthy")
# Everything found in the source will be dumped into this specific folder!
TARGET_CLASS_NAME = "Mosaic"

# 80% for training, 20% for validation
TRAIN_RATIO = 0.8 

def split_dataset():
    print(f"🚀 Gathering all images for exactly: '{TARGET_CLASS_NAME}'...\n")

    train_class_dir = os.path.join(TARGET_FOLDER, 'train', TARGET_CLASS_NAME)
    val_class_dir = os.path.join(TARGET_FOLDER, 'validation', TARGET_CLASS_NAME)
    
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Gather ALL images from all subfolders inside the SOURCE_FOLDER
    all_images = []
    
    # Check if the source folder has subfolders (like "Bacterial Blight", etc.)
    for root, dirs, files in os.walk(SOURCE_FOLDER):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                # Store the exact path to the file
                all_images.append(os.path.join(root, file))

    if not all_images:
        print("❌ Error: No images found in the SOURCE_FOLDER. Check your path!")
        return

    print(f"📸 Found {len(all_images)} total images. Shuffling and splitting...")

    # Shuffle the giant list of images
    random.seed(42)
    random.shuffle(all_images)

    # Calculate the 80/20 split
    split_index = int(len(all_images) * TRAIN_RATIO)
    train_images = all_images[:split_index]
    val_images = all_images[split_index:]

    # Copy to TRAIN folder
    for img_path in train_images:
        file_name = os.path.basename(img_path)
        dst = os.path.join(train_class_dir, file_name)
        # We use a try-except block just in case two files have the exact same name
        try:
            shutil.copy2(img_path, dst)
        except Exception:
            pass 

    # Copy to VALIDATION folder
    for img_path in val_images:
        file_name = os.path.basename(img_path)
        dst = os.path.join(val_class_dir, file_name)
        try:
            shutil.copy2(img_path, dst)
        except Exception:
            pass

    print(f"✅ SUCCESS! Images placed exactly in:")
    print(f"   -> train/{TARGET_CLASS_NAME}/ ({len(train_images)} images)")
    print(f"   -> validation/{TARGET_CLASS_NAME}/ ({len(val_images)} images)")

if __name__ == "__main__":
    split_dataset()