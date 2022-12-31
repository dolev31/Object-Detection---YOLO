import argparse
import cv2
import torch
from segment_utils.smoothing import majority_voting_categorical_segmentation
from segment_utils.segment import single_segmentation, single_img_annotation
from segment_utils.labels import tool_usage_mapper
from segment_utils.evaluation import f1_acc


def segment_video(
        video_file_name,
        model_path,
        video_labels_path=None
):
    video_name = video_file_name.split(".")[0]
    print(f"Inference video: {video_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local').to(device)
    model.max_det = 2
    model.amp = True

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_writer = cv2.VideoWriter(
        filename="results/" + video_name + "_annotated.wmv",
        fourcc=fourcc,
        fps=30.0,
        frameSize=(640, 480)
    )
    video_capture = cv2.VideoCapture(filename=("videos/" + video_file_name))

    if not video_capture.isOpened():
        print("Error opening video stream or file")

    segmentations = {
        "right": [],
        "left": []
    }
    frames = []
    bboxes = []

    # read frames & segment
    while video_capture.isOpened():

        # read next video frame
        retval, image = video_capture.read()

        # stop condition
        if not retval:
            video_capture.release()
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_res = model(image)
        pred, right_label, left_label = single_segmentation(frame_res)

        frames.append(image)
        bbox = [p[0:4] for p in pred]
        bboxes.append(bbox)
        segmentations["right"].append(right_label)
        segmentations["left"].append(left_label)

    # smooth segment predictions
    smoothed_right = majority_voting_categorical_segmentation(
        original_segmentation=segmentations["right"],
        window_size=60
    )
    smoothed_left = majority_voting_categorical_segmentation(
        original_segmentation=segmentations["left"],
        window_size=60
    )

    # write back frames and predictions
    for frame, bbox, right_pred, left_pred in zip(frames, bboxes, smoothed_right, smoothed_left):
        new_frame = single_img_annotation(frame, bbox, right_pred, left_pred)
        video_writer.write(new_frame)

    video_writer.release()

    if video_labels_path:
        # do evaluation

        # load ground truth labels
        with open(video_labels_path + f"/tools_right/{video_name}.txt", "r") as f:
            actual_right_labels_ranges = [line.strip().split(" ") for line in f.readlines()]

        with open(video_labels_path + f"/tools_left/{video_name}.txt", "r") as f:
            actual_left_labels_ranges = [line.strip().split(" ") for line in f.readlines()]

        # by frame
        actual_right_labels = []
        actual_left_labels = []

        for low, high, label in actual_right_labels_ranges:
            low = int(low)
            high = int(high)
            actual_right_labels += ["Right_" + tool_usage_mapper[label]] * (high - low + 1)

        for low, high, label in actual_left_labels_ranges:
            low = int(low)
            high = int(high)
            actual_left_labels += ["Left_" + tool_usage_mapper[label]] * (high - low + 1)

        min_frames = min(len(frames), len(actual_right_labels))
        actual_right_labels = actual_right_labels[:min_frames]
        actual_left_labels = actual_left_labels[:min_frames]
        non_smoothed_right = segmentations["right"][:min_frames]
        non_smoothed_left = segmentations["left"][:min_frames]
        smoothed_right = smoothed_right[:min_frames]
        smoothed_left = smoothed_left[:min_frames]

        # non-smoothed eval
        non_smoothed_f1, non_smoothed_acc = f1_acc(actual_right_labels, actual_left_labels, non_smoothed_right,
                                                   non_smoothed_left)

        # non-smoothed eval
        smoothed_f1, smoothed_acc = f1_acc(actual_right_labels, actual_left_labels, smoothed_right,
                                           smoothed_left)

        print("==== Evaluations ==== \n")
        print("Without Smoothing")
        print("F1:", non_smoothed_f1, "Acc:", non_smoothed_acc)

        print("\nWith Smoothing")
        print("F1:", smoothed_f1, "Acc:", smoothed_acc)


def label_five_videos():
    video_file_names = [
        'P022_balloon1.wmv',
        'P023_tissue2.wmv',
        'P024_balloon1.wmv',
        'P025_tissue2.wmv',
        'P026_tissue1.wmv'
    ]

    model_path = 'models/best.pt'
    video_labels_path = "datasets/HW1_dataset/tool_usage"

    for video_file_name in video_file_names:
        segment_video(video_file_name, model_path, video_labels_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--video_file_name', help='name of video to predict, assumed to be in <root dir>/videos')
    parser.add_argument('--model_path', help='path to pretrained yolo model')
    parser.add_argument('--video_labels_path', default=None, help='path to labels')
    args = parser.parse_args()
    return args.video_file_name, args.model_path, args.video_labels_path


def main():
    video_file_name, model_path, video_labels_path = parse_args()
    segment_video(video_file_name, model_path, video_labels_path)


if __name__ == "__main__":
    label_five_videos()
