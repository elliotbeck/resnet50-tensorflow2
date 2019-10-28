import tensorflow as tf


class DenseNet(tf.keras.Model):
    INPUT_SHAPE = [224, 224]

    def __init__(self, num_classes, densenet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]
        
        self.model = tf.keras.Sequential([
            tf.compat.v1.keras.applications.DenseNet121(include_top=False, 
                weights=densenet_weights, input_shape=in_shape),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dense(num_classes, activation='softmax')
        ])
        self.model.build([None] + self.input_shape + [3])  # Batch input shape.
        

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return DenseNet.INPUT_SHAPE
