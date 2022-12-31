import argparse
import torch
import cv2
from segment_utils.segment import single_segmentation, single_img_annotation


def segment_img(
        img_file_path,
        model_path
):
    img_name = ".".join(img_file_path.split("/")[-1].split(".")[0:-1])
    result_path = r"{}".format(f"results/{img_name}.jpg")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device used:", device)
    model = torch.hub.load('yolov5', 'custom', path=model_path, source='local').to(device)
    model.max_det = 2
    model.amp = True

    img = cv2.imread(img_file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_res = model(img)
    pred, right_label, left_label = single_segmentation(frame_res)
    bbox = [p[0:4] for p in pred]
    new_img = single_img_annotation(img, bbox, right_label, left_label)
    if not cv2.imwrite(result_path, new_img):
        raise Exception("Could not write image")


def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--img_file_path', help='path of image to annotate')
    parser.add_argument('--model_path', help='path to pretrained yolo model')

    args = parser.parse_args()
    return args.img_file_path, args.model_path


def main():
    img_file_path, model_path = parse_args()
    segment_img(img_file_path, model_path)


if __name__ == "__main__":
    main()
