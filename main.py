import os

from zipfile import ZipFile

import cv2
import gdown
import numpy as np
import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator
from src.gui.gui import gui_args
from src.linearReg import Accuracy


PATH_TO_VIDEO = "images/videoplayback_Trim.mp4"


if __name__ == "__main__":
    # Global parameters
    IMAGE_SHAPE = (256, 256)
    IMAGE_CHANNELS = 3
    KEYPOINTS_MODEL_PATH = "tf_pose/models/keypoints/graph_opt.pb"
    
    # Download the models if not downloaded
    # Keypoints
    """
    if os.path.isfile(KEYPOINTS_MODEL_PATH + ".zip") == False:
        gdown.download(
            url = "https://drive.google.com/uc?id=1CnzHB1r6xFAZSc1a0vL-IKkZN6hpFH63",
            output = KEYPOINTS_MODEL_PATH + ".zip"
        )
        # Unzip the files
        with ZipFile(KEYPOINTS_MODEL_PATH + ".zip", "r") as unzip:
            # Extract all the contents of zip file in current directory
            unzip.extractall("models")
    """
    e = TfPoseEstimator(
        KEYPOINTS_MODEL_PATH, 
        target_size=IMAGE_SHAPE)
    
    # Start the GUI
    # args = gui_args()

    """
    print("Camera Number: ", args.camera_number)
    print("Camera On: ", args.camera_on)

    if args.camera_on == True:
        cap = cv2.VideoCapture(0) #args.camera_number)
        while True:
            ret, frame = cap.read()
            frame = cv2.resize(
                frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
            cv2.imshow('Input', frame)
            humans = e.inference(
                frame, 
                resize_to_default=(IMAGE_SHAPE[0] > 0 and IMAGE_SHAPE[1] > 0), 
                upsample_size=4
            )
            print(humans)   
    """

    # Read Image
    
    cap = cv2.VideoCapture(PATH_TO_VIDEO) # args.video_file)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.resize(
                frame, (256, 256), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
            )
            humans = e.inference(
                frame, 
                resize_to_default=(IMAGE_SHAPE[0] > 0 and IMAGE_SHAPE[1] > 0), 
                upsample_size=4
            )
            print(humans)
            acc = Accuracy(humans)
            print(acc)

            ## TODO
            ## work on display video
            ## overlay accuracy on video


