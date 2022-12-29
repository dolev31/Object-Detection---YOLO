from os import listdir
from os.path import isfile, join
import torch
import bbox_visualizer as bbv
import cv2
from tqdm import tqdm
import os

def flip_images():

    files_texts = ["./data/train/images/" + file_name.strip() for file_name in os.listdir("./data/train/images")]

    # train_labels = ["./data/train/labels/" + file_name.strip() for file_name in os.listdir("./data/train/labels")]


    switch = {i: i + 1 if i % 2 == 0 else i - 1 for i in range(8)}

    for image in tqdm(train_files):
        try:
            originalImage = cv2.imread(image)
            # cv2.imshow("bla", originalImage)
            flipHorizontal = cv2.flip(originalImage, 1)
            # cv2.imshow("bla2", flipHorizontal)
            cv2.imwrite(image.replace(".jpg", "_flipped.jpg"), flipHorizontal)
            labels = open(image.replace(".jpg", ".txt").replace("images", 'labels')).readlines()
            new_labels = [line.split(" ") for line in labels]
            to_replace = []
            for line in new_labels:
                shape = originalImage.shape
                label, x, y, w, h = line
                label = str(switch[int(label)])
                x = float(x.strip()) * shape[0]
                y = str(float(y.strip()))
                w = str(float(w.strip()))
                h = str(float(h.strip())) + "\n"
                x = str(abs(shape[0] - x) / shape[0])
                to_replace.append(" ".join([label, x, y, w, h]))
                with open(image.replace(".jpg", "_flipped.txt").replace("images", 'labels'), 'w') as f:
                    for line in to_replace:
                        f.write(line)
        except Exception as e:
            print(e)
            continue


if __name__ == "__main__":
    flip_images()

