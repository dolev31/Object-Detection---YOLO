# Tool Detection & Segmentation In Surgical Room via YOLO5

This repo contains an easy-to-use model for detecting & segmenting tools in both hans of a suturing simulator.  

## Getting Started

### Dependencies

* The dependecies are list via the ruquirments.txt file in the repo.

### Executing program

* Segmenting a video
     * Your video file should be located in a "videos" directory
```
python video.py --video_file_name videos\[your_video_name] --model_path models\[your_model_name]
```

* Segmenting a video with evaluation
     * Your video file should be located in a "videos" directory
     * Your label direrctory should consists two sub-directories: tools_right & tools_left
```
python video.py --video_file_name videos\[your_video_name] --model_path models\[your_model_name] --video_labels_path [path_to_label_dir]
```

* Segmenting an image
     * Your image file should be located in a "images" directory
```
python predict.py --image_file_name images\[your_video_name] --model_path models\[your_model_name]
```

## Authors
* [@Ido Levi](https://github.com/dolev31)
* [@Daniel Yehezkel](https://github.com/daniel-yehezkel)

## Version History
* 0.1
    * Initial Release
