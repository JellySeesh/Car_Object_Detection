from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import*

cap = cv2.VideoCapture("../Videos/cars.mp4") # For Video

model = YOLO('../../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
"dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
"handbag","tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
"baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
"fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
"carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "potted plant", "bed",
"diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
"teddy bear", "hair drier", "toothbrush"
]

mask = cv2.imread("mask.png")

# Tracking
tracker = Sort(max_age=28, min_hits=3, iou=0.3)

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img,mask)

    results = model(imgRegion,stream=True)

    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:

            #bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            #convert the tensor values to actual values which we can use to create a rectanglew
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(x1, y1, x2, y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "car" or currentClass == "truck" or currentClass == "bus" or\
                currentClass == "motorbike" and conf > 0.3:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(0, y1)), scale=0.6, thickness=1, offset=4)#3 n 3 are default values, can change it to 0.7, 1 etc.
                cvzone.cornerRect(img, (x1, y1, w, h), l=9)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1,y1,x2,y2,Id = result
        print(result)

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(0) #0 so that it moves forward only when any key is pressed