import os

from zipfile import ZipFile

import cv2
import time
import gdown
import numpy as np
import tensorflow as tf

from tf_pose.estimator import TfPoseEstimator
from src.gui.gui import gui_args
from src.linearReg import Accuracy


# PATH_TO_VIDEO = "images/videoplayback_Trim.mp4"


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
    args = gui_args()

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
    PATH_TO_VIDEO = args.Video_File
    print(PATH_TO_VIDEO)
    cap = cv2.VideoCapture(PATH_TO_VIDEO) # args.video_file)

    frame_up  = None
    avg_acc = 0
    counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        try:
            if frame == None:
                break
        except:
            pass
        image = cv2.resize(
            frame, (256, 256), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
        humans, humans_full = e.inference(
            image, 
            resize_to_default=(IMAGE_SHAPE[0] > 0 and IMAGE_SHAPE[1] > 0), 
            upsample_size=4
        )
        # print(humans)
        try:
            avg_acc = avg_acc + Accuracy(humans)
        except:
            None # avg_acc = 0.00
        # print(acc)

        # print(frame)

        ## TODO
        ## work on display video
        frame_up = e.draw_humans(frame, humans_full)
        ## overlay accuracy on video

        # sprint(frame)

        cv2.imshow('tf-pose-estimation result', frame_up)
        if cv2.waitKey(1) == 27:
            break
        counter = counter + 1

    avg_acc = avg_acc / counter
    print(avg_acc)
    cv2.putText(frame_up, "Form Accuracy: " + str(avg_acc), (0, frame_up.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow('tf-pose-estimation result', frame_up)
    cv2.waitKey()
    # cv2.destroyAllWindows()
    

