from os import listdir
from os.path import isfile, join
import torch
import bbox_visualizer as bbv
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', source='local')
model.max_det = 8  # maximum number of detections per image
model.amp = True  # Automatic Mixed Precision (AMP) inference

with open("datasets/train.txt") as f:
    train_files = ["datasets/images/" + line.strip() for line in f.readlines()]

with open("datasets/valid.txt") as f:
    valid_files = ["datasets/images/" + line.strip() for line in f.readlines()]

with open("datasets/test.txt") as f:
    test_files = ["datasets/images/" + line.strip() for line in f.readlines()]

tools_right = [f for f in listdir("datasets/tool_usage/tools_right") if
               isfile(join("datasets/tool_usage/tools_right", f))]
tools_left = [f for f in listdir("datasets/tool_usage/tools_left") if isfile(join("datasets/tool_usage/tools_left", f))]

# for img in test_files:
#     results = model(img)
#     image_rgb = cv2.cvtColor(results.render()[0], cv2.COLOR_BGR2RGB)
#     cv2.imshow("image", image_rgb)
#     cv2.waitKey()
#     break
#


## https://learnopencv.com/read-write-and-display-a-video-using-opencv-cpp-python/      ## basic opencv tutorial
## https://github.com/shoumikchow/bbox-visualizer  ## bbox_visualizer git with examples


labels = ['Right_Scissors',
          'Left_Scissors',
          'Right_Needle_driver',
          'Left_Needle_driver',
          'Right_Forceps',
          'Left_Forceps',
          'Right_Empty',
          'Left_Empty']
idx_to_label = {i: inst for i, inst in zip(range(len(labels)), labels)}
class_appearance = {i: 0 for i in range(len(labels))}

cap = cv2.VideoCapture('videos/P022_balloon1.wmv')

# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

running_average = {}

# Read until video is completed
i = 0
while cap.isOpened():
    i += 1
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        # add bounding boxes
        # bbox = [xmin, ymin, xmax, ymax]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_res = model(frame)
        bbox = frame_res.pandas().xyxy[0][['xmin', 'ymin', 'xmax', "ymax"]].astype(int).values.tolist()
        classes = frame_res.pandas().xyxy[0][['confidence', 'name']].values.tolist()
        classes = [f"Class:{label}, Conf: {conf}" for conf, label in classes]
        frame = frame_res.render()[0]
        bbox_labels = [[xmin - 50, ymin, xmax, ymax] for xmin, ymin, xmax, ymax in bbox]
        frame = bbv.add_multiple_labels(frame, classes, bbox, text_bg_color=(0, 255, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
