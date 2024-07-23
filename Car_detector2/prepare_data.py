import os
import shutil
import sklearn
from sklearn.model_selection import train_test_split

# Path to your dataset directory
dataset_root = "C:\\Users\\Shresth.Aggarwal\\PycharmProjects\\object_detection\\Car_detector2\\data"

# Path to training images directory
training_images_dir = os.path.join(dataset_root, "images")

labels_dir = os.path.join(dataset_root, "labels")

# Get list of all image files
all_images = os.listdir(training_images_dir)

# Split into train and validation sets
train_images, val_images = train_test_split(all_images, test_size=0.2, random_state=42, shuffle=True)

# Define paths for train and validation directories
train_dir = os.path.join(dataset_root, "train_split")
val_dir = os.path.join(dataset_root, "val_split")

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Move images to respective directories
for img in train_images:
    shutil.move(os.path.join(training_images_dir, img), os.path.join(train_dir, img))

for img in val_images:
    shutil.move(os.path.join(training_images_dir, img), os.path.join(val_dir, img))
