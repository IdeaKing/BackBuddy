"""
Keypoints dataset.py

Reference: https://keras.io/examples/vision/keypoint_detection/

"""
import os
import cv2
import json
import shutil
import numpy as np

import tensorflow as tf
import albumentations as A

from datetime import datetime

class KeypointsDataset:
    def __init__(
        self,
        dataset_dir,
        image_shape=(256, 256),
        image_dir="images",
        label_dir="labels",
        shuffle_size=16,
        batch_size=10,
        augment=False,
        annotation_file=None,
        create_keypoints_files=False,
        maximum_people_per_image=10
    ):
        """
        Dataset for keypoints generation.
        :param configs (class): Configurations for the dataset
        :return:
        """
        self.dataset_dir = dataset_dir
        self.image_shape = image_shape
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.shuffle_size = shuffle_size
        self.batch_size = batch_size
        self.augment_dataset = augment
        self.path_to_annotation_file = annotation_file
        self.maximum_people_per_image = maximum_people_per_image

        print(datetime.now(), " Checking if keypoints required...")
        if create_keypoints_files==True:
            # Create a label dir if needed
            if os.path.isdir(
                os.path.join(
                    self.dataset_dir,
                    self.label_dir
                )
            ) == False:
                os.mkdir(
                    os.path.join(
                        self.dataset_dir,
                        self.label_dir
                    )
                )
            print(datetime.now(), " Saving keypoints...")
            self.save_coco_keypoints()
        
        print(datetime.now(), " Checking if training file is required...")
        if os.path.isfile(
            os.path.join(
                self.dataset_dir,
                "train.txt"
            )
        ) == False:
            print(datetime.now(), " Creating training file...")
            self.create_training_file()

    def read_dataset(
        self,
        path
    ):
        path = tf.compat.as_str_any(path)
        image_path = os.path.join(
            self.dataset_dir,
            self.image_dir,
            path + ".jpg"
        )
        label_path = os.path.join(
            self.dataset_dir,
            self.label_dir,
            path + ".json"
        )
        image = self.read_image(image_path)
        keypoints = self.read_coco_keypoints(label_path)
        if self.augment_dataset:
            image, keypoints = self.augment(image, keypoints)
        return image, keypoints


    def read_image(
        self,
        path
    ):
        '''
        path = tf.compat.as_str_any(path)
        path = os.path.join(
            self.dataset_dir,
            self.image_dir,
            path + ".jpg"
        )
        '''
        # print("------------------- path -------------------------")
        # print(path)
        image = cv2.imread(path)
        image = cv2.resize(
            image,
            self.image_shape
        )
        """
        image = tf.io.decode_image(path)
        image = tf.image.resize(
            image,
            self.image_shape
        )
        """
        image = image/255.

        image = np.array(image, dtype = np.float32)
        return image


    def read_coco_keypoints(
        self,
        path_to_keypoints
    ):
        """
        Parses the file with the keypoints.
        :param path_to_keypoints (str): Path to the .json file for a specific image
        """
        '''
        # Read the file to the key points
        path_to_keypoints = tf.compat.as_str_any(
            path_to_keypoints
        )
        path_to_keypoints = os.path.join(
            self.dataset_dir,
            self.label_dir,
            path_to_keypoints + ".json"
        )
        '''
        with open(path_to_keypoints, "r") as f:
            org_dataset = json.load(f)
            assert type(org_dataset)==dict
        
        keypoints_dataset = org_dataset["keypoints"]
        # There is a maximum 10 of ten people per keypoint
        # dataset = np.zeros([self.maximum_people_per_image, 57])
        dataset = []

        for x in range(len(keypoints_dataset)):
            _temp = []
            if keypoints_dataset[x] != None:
                for z in range(len(keypoints_dataset[x])):
                    z = z+1
                    if z % 3 == 0: 
                        pass
                    elif z % 3 == 2:
                        _temp.append(float(int(keypoints_dataset[x][z-1])/int(org_dataset["height"])))
                    elif z % 3 == 1:
                        _temp.append(float(int(keypoints_dataset[x][z-1])/int(org_dataset["width"])))
                dataset.append(_temp)
        for x in range(self.maximum_people_per_image - len(dataset)):
            dataset.append([0] * 38)

        dataset = np.array(dataset, dtype = np.float32)


        #print("--------------- dataset --------------------", dataset)
        return dataset


    def read_training_file(
        self
    ):
        """
        Reads the training file line by line to create the file list
        :returns file_list:
        """


    def create_training_file(
        self
    ):
        """
        Creates a training file with all of the images for training.
        """
        image_file_dir = os.path.join(
            self.dataset_dir,
            self.image_dir
        )
        training_file_path = os.path.join(
            self.dataset_dir,
            "train.txt"
        )
        if os.path.isfile(training_file_path):
            print(
                datetime.now(), 
                " The training file (training.txt) exists in dataset directory."
            )
        else:
            image_file_list = os.listdir(image_file_dir)
            with open(training_file_path, "w") as writer:
                for image_file in image_file_list:
                    image_file = image_file[:-4]
                    writer.write(image_file)
                    writer.write("\n")

    def augment(
        self,
        image,
        label
    ):
        transform = A.Compose(
            [A.RandomCrop(width=330, height=330),
             A.RandomBrightnessContrast(p=0.2)], 
            keypoint_params=A.KeypointParams(format='xy')
        )
        transformed_data = transform(image=image, keypoints=label)
        image, label = transformed_data["image"], transformed_data["keypoints"]
        return image, label


    def save_coco_keypoints(
        self
    ):
        """
        Function to write a separate label file for each image.
        :return:
        """
        # Read the file to the key points
        path_to_annotation_file = tf.compat.as_str_any(
            self.path_to_annotation_file
        )
        with open(path_to_annotation_file, "r") as f:
            ground_truth = json.load(f)
            assert type(ground_truth)==dict
        # Read the COCO format file
        for label in ground_truth["images"]:
            annotation_dict = {}
            annotation_dict["image_id"] = label["image_id"]
            annotation_dict["height"] = label["height"]
            annotation_dict["width"] = label["width"]
            keypoints_list = []
            for keypoints in label["annotations"]:
                keypoints_list.append(keypoints["keypoints"])
            annotation_dict["keypoints"] = keypoints_list
            # Save the labels to directory
            save_path = os.path.join(
                self.dataset_dir,
                self.label_dir, 
                annotation_dict["image_id"] + ".json"
            )
            with open(save_path, "w") as label_file:
                label_file.write(json.dumps(annotation_dict))


    def create_dataset(
        self
    ):
        """
        Creates a tf.data.Dataset for keypoint training.
        :param file_name_list (str): A list with all of the file names 
        :returns:
        """
        print(datetime.now(), " Compiling the TF Dataset...")
        file_name_path = os.path.join(
            self.dataset_dir,
            "train.txt"
        )
        file_name_lines = open(file_name_path, "r")
        file_name_lines = file_name_lines.readlines()
        file_name_list = [line.strip() for line in file_name_lines]
        file_name_ds = tf.data.Dataset.from_tensor_slices(
            file_name_list,
            name = "file_name_dataset"
        )
        dataset = file_name_ds.map(
            lambda x: 
                tf.numpy_function(
                    self.read_dataset,
                    inp=[x],
                    Tout=[tf.float32, tf.float32],
                    name="read_dataset"
                ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            name = "full_dataset"
        )
        if self.shuffle_size > 0:
            dataset = dataset.shuffle(
                self.shuffle_size
            )
        if self.batch_size > 0:
            dataset = dataset.batch(
                self.batch_size
            )
        dataset = dataset.prefetch(
            buffer_size=tf.data.experimental.AUTOTUNE
        )
        return dataset