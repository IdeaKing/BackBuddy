import tensorflow as tf
import numpy as np
import cv2

def keypoints_pipeline(
    images,
    model,
    process_batch_size=10,
    image_shape=(256, 256),
    image_channels=3,
    visualize=False
):  
    """
    Pipeline for Keypoints inference.
    :returns keypoints, in float, not pixels:
    """
    image_arr = np.array(
        (process_batch_size, image_shape[0], image_shape[1], image_channels))

    counter = 0
    for image in images:
        image = cv2.resize(
            image, 
            (image_shape[0], image_shape[1], image_channels)
        )
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
        image_arr[counter] = image

    outputs = model(image_arr, training=False)
    outputs = outputs.numpy()
    return outputs