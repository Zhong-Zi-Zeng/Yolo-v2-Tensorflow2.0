import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

class CBR(Layer):
    def __init__(self, filters, kernel):
        super().__init__()
        self.cbr = Sequential([
            Conv2D(filters, kernel, padding='same', use_bias=False),
            BatchNormalization(),
            ReLU()
        ])

    def call(self, inputs, *args, **kwargs):
        return self.cbr(inputs)


class Darknet19(Model):
    def __init__(self, k, num_classes):
        super().__init__()
        # output dim (13, 13, k, (5 + num_classes))

        self.darknet_19 = Sequential([
            # Model 1
            CBR(32, 3),
            MaxPooling2D(strides=2),

            # Model 2
            CBR(64, 3),
            MaxPooling2D(strides=2),

            # Model 3
            CBR(128, 3),
            CBR(64, 1),
            CBR(128, 3),
            MaxPooling2D(strides=2),

            # Model 4
            CBR(256, 3),
            CBR(128, 1),
            CBR(256, 3),
            MaxPooling2D(strides=2),

            # Model 5
            CBR(512, 3),
            CBR(256, 1),
            CBR(512, 3),
            CBR(256, 1),
            CBR(512, 3),
            MaxPooling2D(strides=2),

            # Model 6
            CBR(1024, 3),
            CBR(512, 1),
            CBR(1024, 3),
            CBR(512, 1),
            CBR(1024, 3),
            CBR(1024, 3),
            CBR(1024, 3),

            # Output
            Conv2D(k * (5 + num_classes), 1, padding='same'),
            Reshape((13, 13, k, 5 + num_classes))
        ])

    def call(self, inputs, training=None, mask=None):
        return self.darknet_19(inputs)

