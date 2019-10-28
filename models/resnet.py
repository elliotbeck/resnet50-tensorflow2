import tensorflow as tf


class ResNet50(tf.keras.Model):
    INPUT_SHAPE = [224, 224]

    def __init__(self, num_classes, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        self.model = tf.keras.models.Sequential([
            tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    weights= resnet_weights, input_shape=in_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return ResNet50.INPUT_SHAPE