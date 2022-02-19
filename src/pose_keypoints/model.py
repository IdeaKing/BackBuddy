import tensorflow as tf

import backbone as backbone

def build_model(
    input_shape=(256,256),
    channels = 3,
    output = 38,
    backbone="101"
):
    """
    Builds the model for keypoints.
    """
    backbone = backbone.ResNet(
        model_type=backbone,
        input_shape=(input_shape[0], input_shape[1], channels)
    ).__call__()
    backbone.trainable = True

    inputs = tf.keras.layers.Input(
        (input_shape[0], input_shape[1], channels))
    
    x = backbone(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.SeparableConv2D(
        output, kernel_size=5, strides=1, activation="relu")(x)
    x = tf.keras.layers.SeparableConv2D(
        output, kernel_size=3, strides=1, activation="relu")(x)
    outputs = tf.keras.layers.SeparableConv2D(
        output, kernel_size=2, strides=1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="keypoint_detector")

    return model