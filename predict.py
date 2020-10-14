import tensorflow as tf
import config
import numpy as np
import pathlib
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
from config import image_height, image_width, channels, train_dir


def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.load_weights(config.save_model_dir)
    model.summary()
    return model
    
def load_image_input(img_path):
    #load image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(image_height, image_width))
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_img = np.expand_dims(img_arr, axis=0)
    input_img = tf.keras.utils.normalize(input_img)
    return input_img
    
def get_label(prediction):
    data_root = pathlib.Path(train_dir)
    label_names = sorted(item.name for item in data_root.glob('*/'))
    return label_names[prediction]
    
    
if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # create model
    model = get_model()
    # change path to your image
    # make prediction
    prediction = model.predict(load_image_input(path))
    # print prediction
    print("Prediction:",get_label(np.argmax(prediction[0])), "\nAccuracy:", prediction[0][np.argmax(prediction)])
