from os import listdir
from os.path import isfile, join
import torch
import bbox_visualizer as bbv
import cv2
from tqdm import tqdm
import os


def flip_images():
    train_dataset = ["./data/train/images/" + file_name.strip() for file_name in os.listdir("./data/train/images")]

    label_map = {0: 0, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7}

    for image in tqdm(train_dataset):
        try:
            org_image = cv2.imread(image)
            flipped_image = cv2.flip(org_image, 1)
            # Write the flipped image
            cv2.imwrite(image.replace(".jpg", "_flipped.jpg"), flipped_image)
            labels = open(image.replace(".jpg", ".txt").replace("images", 'labels')).readlines()
            aug_labels = [line.split(" ") for line in labels]
            replace = []
            for line in aug_labels:
                # shape = org_image.shape
                label, x, y, w, h = line
                label = str(label_map[int(label)])
                # x = float(x.strip()) * shape[0]
                y = str(float(y.strip()))
                w = str(float(w.strip()))
                h = str(float(h.strip())) + "\n"
                # x = str(abs(shape[0] - x) / shape[0])
                replace.append(" ".join([label, x, y, w, h]))
                with open(image.replace(".jpg", "_flipped.txt").replace("images", 'labels'), 'w') as f:
                    for line in replace:
                        f.write(line)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    flip_images()
