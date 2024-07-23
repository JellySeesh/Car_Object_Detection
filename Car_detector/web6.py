import os
import cv2
import streamlit as st
from ultralytics import YOLO

# Load the model and set the threshold
model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'best.pt')
model = YOLO(model_path)
threshold = 0.5

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

# Create a Streamlit app
st.set_page_config(page_title="Object Detection App", layout="wide")
st.title("Object Detection App")

# Allow user to select the directory or a file
st.subheader("Select Image Source")
image_source = st.radio("Select image source:", ("Directory", "File"), horizontal=True)

if image_source == "Directory":
    # Allow user to select the directory
    images_dir = st.text_input("Enter the directory path:", "testing_images")
    if os.path.isdir(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    else:
        st.warning("No directory selected.")
        image_files = []
else:
    # Allow user to select a file
    image_file = st.file_uploader("Choose an image file", type=["jpg"])
    if image_file is not None:
        with open(os.path.join("testing_images", image_file.name), "wb") as f:
            f.write(image_file.getbuffer())
        image_files = [image_file.name]
        images_dir = "testing_images"
    else:
        st.write("No image file selected.")
        image_files = []

# Process the selected images
st.subheader("Object Detection Results")
for image_file in image_files:
    image_path = os.path.join(images_dir, image_file)
    
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
        st.write(f'Object: {class_name}, Score: {score:.2f}')
        
        if score > threshold:
            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
            
    # Optionally, save annotated image
    output_image_path = os.path.join('output', f'annotated_{image_file}')
    cv2.imwrite(output_image_path, frame)

    # Display the image
    st.image(frame, caption=f'Processed: {image_file}')   # can add ", use_column_width=True"

print('Processing complete.')