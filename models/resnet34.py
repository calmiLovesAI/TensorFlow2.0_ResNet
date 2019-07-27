import tensorflow as tf
from models.residual_block import BasicBlock, build_res_block
from config import NUM_CLASSES, image_height, image_width, channels


class ResNet34(tf.keras.layers.Layer):

    def __init__(self, num_classes=NUM_CLASSES):
        super(ResNet34, self).__init__()

        self.preprocess = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(pool_size=2,
                                      strides=1,
                                      padding='same')
        ])

        self.layer1 = build_res_block(filter_num=64,
                                      blocks=3)
        self.layer2 = build_res_block(filter_num=128,
                                      blocks=4,
                                      stride=2)
        self.layer3 = build_res_block(filter_num=256,
                                      blocks=6,
                                      stride=2)
        self.layer4 = build_res_block(filter_num=512,
                                      blocks=3,
                                      stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_classes)

    def call(self, inputs, training=None):
        pre = self.preprocess(inputs)
        l1 = self.layer1(pre)
        l2 = self.layer1(l1)
        l3 = self.layer1(l2)
        l4 = self.layer1(l3)
        avgpool = self.avgpool(l4)
        out = self.fc(avgpool)

        return out