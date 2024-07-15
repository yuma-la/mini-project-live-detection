import cv2
import numpy as np
import requests

# URLs for YOLO files
cfg_url = "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov3.cfg"
weights_url = "https://pjreddie.com/media/files/yolov3.weights"
names_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"

# Function to download files
def download_file(url, file_name):
    response = requests.get(url, stream=True)
    with open(file_name, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)

# Download YOLO files
download_file(cfg_url, "yolov3.cfg")
download_file(weights_url, "yolov3.weights")
download_file(names_url, "coco.names")

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load webcam
cap = cv2.VideoCapture(0)

# Set resolution to lower to reduce lag
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    height, width, channels = frame.shape

    # Detecting objects every 3rd frame to reduce lag
    if cap.get(cv2.CAP_PROP_POS_FRAMES) % 3 == 0:
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Menunjukan detail informasi pada layar
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 0:  # class_id 0 is 'person'
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            color = (0, 255, 0) if label == "mask" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)
    if key == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()



#
