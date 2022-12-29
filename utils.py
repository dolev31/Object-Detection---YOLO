from os import listdir
from os.path import isfile, join
import torch
import bbox_visualizer as bbv
import glob
import os
import cv2
from tqdm import tqdm


class Utils:
    @staticmethod
    def collect_files(dataset_name: str):
        # Define the datasets and image paths
        train_path = "datasets/train/images/*"
        valid_path = "datasets/valid/images/*"
        test_path = "datasets/test/images/*"
        if dataset_name == 'all':
            images_path = glob.glob(train_path) + glob.glob(valid_path) + glob.glob(test_path)
        elif dataset_name == 'train':
            images_path = glob.glob(train_path)
        elif dataset_name == 'valid':
            images_path = glob.glob(valid_path)
        elif dataset_name == 'test':
            images_path = glob.glob(test_path)

    @staticmethod
    def dataset_augmentation(datasets_to_augmentation: str):

        images_path = self.collect_files(datasets_to_augmentation)

        # Create a mapping for the labels
        switch = {i: i + 1 if i % 2 == 0 else i - 1 for i in range(8)}
        matching_lb_dict = {0: 0, 1: 2, 2: 3, 3: 5, 4: 6, 5: 7}

        # Iterate through each image path
        for image_path in tqdm(images_path):
            # Initialize empty lists for the label replacements
            lb_replace, lb_fix = [], []

            try:
                # Read the image and flip it
                image = cv2.imread(image_path)
                flipped_image = cv2.flip(image, 1)

                # Save the flipped image
                cv2.imwrite(image_path.replace(".jpg", "_flipped.jpg"), flipped_image)

                # Read the labels for the image
                with open(image_path.replace(".jpg", ".txt").replace("images", 'labels')) as f:
                    labels = f.readlines()

                # Split the labels into a list of lists
                ground_truth = [line.split(" ") for line in labels]

                # Iterate through each label
                for gt in ground_truth:
                    # Get the shape of the image and the label information
                    shape = image.shape
                    label, x, y, w, h = gt
                    label = str(matching_lb_dict[int(label)])
                    fixed_label = str(switch[int(label)])

                    # Calculate the new x coordinate for the label
                    new_x = float(x.strip()) * shape[0]
                    new_x = str(abs(shape[0] - new_x) / shape[0])

                    # Assemble the modified label text
                    modified_label = " ".join([fixed_label, new_x, y, w, h])
                    lb_replace.append(modified_label)
                    lb_fix.append(" ".join([label, x, y, w, h]))

                # Write the modified labels to the flipped image file
                with open(image_path.replace(".jpg", "_flipped.txt").replace("images", 'labels'), 'w') as f:
                    for gt in lb_replace:
                        f.write(gt)

                # Write the original labels to the original image file
                with open(image_path.replace(".jpg", ".txt").replace("images", 'labels'), 'w') as f:
                    for gt in lb_fix:
                        f.write(gt)

            except Exception as e:
                print(e)
                continue


if __name__ == "__main__":
    utils_object = Utils()
    utils_object.dataset_augmentation('all')  # change to train, valid or test for augmentation
