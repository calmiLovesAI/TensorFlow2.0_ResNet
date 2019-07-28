import tensorflow as tf
from models.residual_block import build_res_block_2
from config import NUM_CLASSES

class ResNet50(tf.keras.Model):
    def __init__(self, inputs, num_classes=NUM_CLASSES):
        super(ResNet50, self).__init__()

        self.preprocess = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(7, 7),
                                   strides=2,
                                   padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation(tf.keras.activations.relu),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                      strides=2)
        ])

        self.layer1 = build_res_block_2(filter_num=64,
                                        blocks=3)
        self.layer2 = build_res_block_2(filter_num=128,
                                        blocks=4,
                                        stride=2)
        self.layer3 = build_res_block_2(filter_num=256,
                                        blocks=6,
                                        stride=2)
        self.layer4 = build_res_block_2(filter_num=512,
                                        blocks=3,
                                        stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=num_classes, activation=tf.keras.activations.softmax)

    def call(self, inputs):
        pre = self.preprocess(inputs)
        l1 = self.layer1(pre)
        l2 = self.layer1(l1)
        l3 = self.layer1(l2)
        l4 = self.layer1(l3)
        avgpool = self.avgpool(l4)
        out = self.fc(avgpool)

        return out

