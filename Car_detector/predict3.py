import os

from ultralytics import YOLO
import cv2

IMAGES_DIR = os.path.join('.', 'testing_images')
OUTPUT_DIR = os.path.join('.', 'output')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to YOLO model weights file
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')

# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.3

# List all image files in the directory
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]

# Process each image
for image_file in image_files:
    image_path = os.path.join(IMAGES_DIR, image_file)
    
    try:
        # Read image using OpenCV
        frame = cv2.imread(image_path)
        H, W, _ = frame.shape
        
        # Perform object detection
        results = model(frame)[0]

        # Process detection results
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            # Print class name and score
            class_name = model.names[int(class_id)]
            print(f'Object: {class_name}, Score: {score:.2f}')

            if score > threshold:
                # Draw bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Optionally, save annotated image
        output_image_path = os.path.join(OUTPUT_DIR, f'annotated_{image_file}')
        cv2.imwrite(output_image_path, frame)

        print(f'Processed: {image_file}')

    except Exception as e:
        print(f'Error processing {image_file}: {str(e)}')

print('Processing complete.')        