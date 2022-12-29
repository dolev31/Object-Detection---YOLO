
# !pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="UKAne430cjV8yrABlkf8")
project = rf.workspace("technion-vb9eo").project("object_detection_sergical_tools")
dataset = project.version(1).download("yolov5")