import os
from ultralytics import YOLO
import cv2

IMAGES_DIR = os.path.join('.', 'testing_images')
OUTPUT_DIR = os.path.join('.', 'output')

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Path to YOLO model weights file
model_path = os.path.join('.', 'runs', 'detect', 'train16', 'weights', 'best.pt')

# Load the model
model = YOLO(model_path)  # load a custom model

threshold = 0.3

# List all image files in the directory
image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith('.jpg')]

# Process each image
for image_file in image_files:
    image_path = os.path.join(IMAGES_DIR, image_file)
    
    # Read image using OpenCV
    frame = cv2.imread(image_path)
    
    if frame is None:
        print(f'Error reading image: {image_path}')
        continue
    
    H, W, _ = frame.shape
    
    # Perform object detection
    results = model(frame)[0]

    # Check if results are returned
    if results.boxes.data is None:
        print(f'No objects detected in {image_file}')
        continue

    # Process detection results
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        class_name = model.names[int(class_id)]

        if score > threshold:
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            
            # Prepare text
            label = f'{class_name.upper()} {score:.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.3
            font_color = (0, 255, 0)  # Green
            font_thickness = 3
            
            # Calculate text size
            text_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            text_width, text_height = text_size
            
            # Calculate text position
            text_x = int(x1)
            text_y = int(y1 - 10 if y1 - 10 > text_height else y1 + text_height)
            
            # Draw text on image
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
            
            # Debug output
            print(f'Drawing text: "{label}" at ({text_x}, {text_y})')

    # Save annotated image
    output_image_path = os.path.join(OUTPUT_DIR, f'annotated_{image_file}')
    cv2.imwrite(output_image_path, frame)
    print(f'Processed: {image_file}')

print('Processing complete.')
