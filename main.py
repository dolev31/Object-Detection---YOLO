from os import listdir
from os.path import isfile, join
import torch
import bbox_visualizer as bbv
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

model = torch.hub.load('yolov5', 'custom', path='yolov5/best.pt', source='local')
model.max_det = 2  # maximum number of detections per image
model.amp = True  # Automatic Mixed Precision (AMP) inference

with open("datasets/train.txt") as f:
    train_files = ["datasets/images/" + line.strip() for line in f.readlines()]

with open("datasets/train.txt") as f:
    train_labels = ["datasets/labels/" + line.strip() for line in f.readlines()]

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

left_running_average = []
right_running_average = []

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
        classes = [f"{label}, {round(conf, 2)}" for conf, label in classes]
        frame = bbv.draw_multiple_rectangles(frame, bbox, bbox_color=(0, 255, 0))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for i, (x, y, w, h) in enumerate(bbox):
            org = (x - 10, y - 10)
            color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            fontScale = 0.5
            image = cv2.putText(frame, classes[i], org, font, fontScale, color, thickness, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        image = cv2.putText(frame, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
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

# font
# font = cv2.FONT_HERSHEY_SIMPLEX
#
# # org
# org = (50, 50) #It is the coordinates of the bottom-left corner of the text string in the image. The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
#
# # fontScale
# fontScale = 1 #Font scale factor that is multiplied by the font-specific base size.
#
# # Blue color in BGR
# color = (255, 0, 0)
#
# # Line thickness of 2 px
# thickness = 2
# text_to_print =
# # Using cv2.putText() method
# image = cv2.putText(image, text_to_print, org, font,
#                     fontScale, color, thickness, cv2.LINE_AA)
