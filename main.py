import pyforms
import tensorflow as tf

from src.GUI.player import ComputerVisionAlgorithm

# Global parameters
IMAGE_SHAPE = (256, 256)
IMAGE_CHANNELS = 3

"""
KEYPOINTS_MODEL_PATH = "models/keypoints"
CLASSIFIER_MODEL_PATH = "models/classifier"

KEYPOINTS_MODEL = tf.keras.models.load_model(
    KEYPOINTS_MODEL_PATH
)
CLASSIFIER_MODEL = tf.keras.models.load_model(
    CLASSIFIER_MODEL_PATH
)
"""
GUI = ComputerVisionAlgorithm(
    keypoints_model=KEYPOINTS_MODEL,
    classifier_model=CLASSIFIER_MODEL
)
if __name__ == "__main__":
    pyforms.start_app(GUI)