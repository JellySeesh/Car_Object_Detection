from ultralytics import YOLO

model = YOLO('../Yolo-Weights/yolov8l.pt') #n for nano weights
results = model("Images/3.png", show=True)
#to stop the image
import cv2
cv2.waitKey(0)