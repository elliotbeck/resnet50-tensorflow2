import tensorflow as tf
import json

class ResNet50(tf.keras.Model):
    INPUT_SHAPE = [224, 224]

    def __init__(self, num_classes, resnet_weights, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        in_shape = self.input_shape + [3]

        # self.model = tf.keras.models.Sequential([
        #     tf.keras.applications.resnet50.ResNet50(include_top=False,
        #                                             weights= resnet_weights, input_shape=in_shape),                                      
        #     tf.keras.layers.Flatten(),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dense(512, activation='relu'),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dense(128, activation='relu'),
        #     tf.keras.layers.Dropout(0.5),
        #     tf.keras.layers.BatchNormalization(),
        #     # tf.keras.layers.Dense(64, activation='relu'),
        #     # tf.keras.layers.Dropout(0.5),
        #     # tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Dense(num_classes, activation='softmax')
        # ])

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.applications.resnet50.ResNet50(include_top=False,
                                                    weights= resnet_weights, input_shape=in_shape))
        # for layer in self.model.layers[:-4]:
        #     layer.trainable = False 
        self.model.add(tf.keras.layers.GlobalAveragePooling2D())                                    
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(256, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(128, activation='relu'))
        self.model.add(tf.keras.layers.Dropout(0.5))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        self.model.build([None] + self.input_shape + [3])  # Batch input shape.

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training, mask)

    @property
    def input_shape(self):
        return ResNet50.INPUT_SHAPE