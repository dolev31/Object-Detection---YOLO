import torch
import cv2


model = torch.hub.load('yolov5s', 'custom', source='local', weights='yolov5/')

model.max_det = 8  # maximum number of detections per image
model.amp = True  # Automatic Mixed Precision (AMP) inference



# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()