import tensorflow as tf


class ResNet:
    def __init__(
        self, 
        model_type, 
        input_shape,
        transfer_learning="imagenet",
        include_top=False,
        pooling=None,
        classes=None,
        classifier_activation=False
    ):
        """
        Builds the specified ResNet Backbone Model.

        :param model_type (str): Can be "50", "101", "152"
        :param input_shape (tuple): Tuple of the image shape (width, height, channels)
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.transfer_learning = transfer_learning
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.classifer_activatation = classifier_activation


    def __call__(self):
        if self.model_type == "50":
            return tf.keras.applications.resnet.ResNet50(
                include_top=self.include_top,
                weights=self.transfer_learning,
                input_shape=self.input_shape,
                pooling=self.pooling,
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "101":
            return tf.keras.applications.resnet.ResNet101(
                include_top=self.include_top,
                weights=self.transfer_learning,
                input_shape=self.input_shape,
                pooling=self.pooling,
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "152":
            return tf.keras.applications.resnet.ResNet152(
                include_top=self.include_top,
                weights=self.transfer_learning,
                input_shape=self.input_shape,
                pooling=self.pooling,
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )


class EfficientNet:
    def __init__(
        self, 
        model_type, 
        input_shape,
        transfer_learning="imagenet",
        include_top=False,
        pooling=None,
        classes=None,
        classifier_activation=False
    ):
        """
        Builds the specified EfficientNet Backbone Model.

        :param model_type (str): Can be "B0", "B1", "B2", "B3", "B4", B5", "B6", "B7"
        :param input_shape (tuple): Tuple of the image shape (width, height, channels)
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.transfer_learning = transfer_learning
        self.include_top = include_top
        self.pooling = pooling
        self.classes = classes
        self.classifer_activatation = classifier_activation


    def __call__(self):
        if self.model_type == "B0":
            return tf.keras.applications.efficientnet.EfficientNetB0(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "B1":
            return tf.keras.applications.efficientnet.EfficientNetB1(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "B2":
            return tf.keras.applications.efficientnet.EfficientNetB2(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "B3":
            return tf.keras.applications.efficientnet.EfficientNetB3(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "B4":
            return tf.keras.applications.efficientnet.EfficientNetB4(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "B5":
            return tf.keras.applications.efficientnet.EfficientNetB5(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "B6":
            return tf.keras.applications.efficientnet.EfficientNetB6(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )
        elif self.model_type == "B7":
            return tf.keras.applications.efficientnet.EfficientNetB7(
                include_top=self.include_top, 
                weights=self.transfer_learning, 
                input_tensor=None,
                input_shape=self.input_shape, 
                pooling=self.pooling, 
                classes=self.classes,
                classifier_activation=self.classifer_activatation
            )