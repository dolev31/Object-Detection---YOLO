# Tool Detection & Segmentation In Surgical Room via YOLO5

This repo contains an easy-to-use model for detecting & segmenting tools in both hans of a suturing simulator.  

## Description

An in-depth paragraph about your project and overview of use.

## Getting Started

### Dependencies

* The dependecies are list via the ruquirments.txt file in the repo.

### Executing program

* Segmenting a video
```
python video.py --video_file_name videos\[your_video_name] --model_path models\[your_model_name]
```

* Segmenting a video with evaluation
```
python video.py --video_file_name videos\[your_video_name] --model_path models\[your_model_name] --video_labels_path [path_to_label_dir]
```

* Segmenting an image
```
python predict.py --image_file_name images\[your_video_name] --model_path models\[your_model_name]
```

## Authors
* [@Ido Levi](https://github.com/dolev31)
* [@Daniel Yehezkel](https://github.com/daniel-yehezkel)

## Version History
* 0.1
    * Initial Release
