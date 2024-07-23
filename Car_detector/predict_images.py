import os
import cv2
from ultralytics import YOLO

IMAGES_DIR = os.path.join('.', 'photos')  # Path to your directory containing images
OUTPUT_DIR = os.path.join('.', 'output')  # Output directory for annotated images or video

# Ensure the output directory exists, create if necessary
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize YOLO model
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
model = YOLO(model_path)  # Load your custom YOLO model

# Threshold for object detection confidence
threshold = 0.5

# Process each image in the directory
for filename in os.listdir(IMAGES_DIR):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Adjust based on your image formats
        image_path = os.path.join(IMAGES_DIR, filename)
        image = cv2.imread(image_path)

        # Perform object detection on the image
        results = model(image)[0]

        # Iterate over detected objects and draw bounding boxes
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(image, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Save annotated image to the output directory
        output_image_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_out.jpg")
        cv2.imwrite(output_image_path, image)

# Optionally, you can create a video from the annotated images
# VideoWriter parameters
fps = 30  # Adjust as needed
output_video_path = os.path.join(OUTPUT_DIR, 'output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (image.shape[1], image.shape[0]))

# Iterate through saved annotated images and write to video
for filename in os.listdir(OUTPUT_DIR):
    if filename.endswith('_out.jpg'):
        image_path = os.path.join(OUTPUT_DIR, filename)
        image = cv2.imread(image_path)
        out_video.write(image)

# Release video writer and cleanup
out_video.release()
cv2.destroyAllWindows()

print(f"Annotations saved to {OUTPUT_DIR}")
print(f"Video created: {output_video_path}")
