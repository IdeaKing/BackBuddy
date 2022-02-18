# Todos for Blairhacks

## General Notes

1. AI will run on greyscale images to preserve computational resources
2. Inputs for the AI model will be either (128, 128) or (256, 256)
3. All keypoints must be scaled according to the input dimensions to ensure cross-compatibility
    1. Scaling formula: scaled-keypoint-x = unscaled-keypoint-x/image-width & scaled-keypoint-y = unscaled-keypoint-y/image-height

## LSTM Training

1. Data stuff
    1. Scrape the data from YouTube using YouTubeDL
    2. Break the video into staggered frames:
        1. For a 60 second clip with 60 fps (total 3600 frames) break down into 2 fps (total 120 fps) then break them into 10 frame intervals and **label each 10 frame intervals with [Good Form, Bad Form, etc...]**
            1. Each 10 frame interval should be batched like this: [10 frames, image-width, image-height, 1]
        2. This module will be used in the production backend
    3. Run the batched 10 frame video feed through the Keypoints Algorithm for input training data
        1. Save the keypoints into datatype of choice - numpy, .json, .txt, etc.
            1. If using .json, can save as a Python dict.
2. LSTM Stuff
    1. Frank - you do you
    2. The outputs should be a class label

## Video feed

1. Video should contain the keypoints on major body parts
