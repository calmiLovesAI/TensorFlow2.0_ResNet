from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models import resnet50, resnet101, resnet152, resnet34
import config
from prepare_data import get_datasets


def get_model():
    model = resnet50.ResNet50()
    if config.model == "resnet34":
        model = resnet34.ResNet34()
    if config.model == "resnet101":
        model = resnet101.ResNet101()
    if config.model == "resnet152":
        model = resnet152.ResNet152()


    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
                  metrics=[tf.keras.metrics.Accuracy()])

    return model


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # get the original_dataset
    # train_dataset, test_dataset, train_count, test_count = get_datasets()
    train_generator, valid_generator, test_generator, train_num, valid_num, test_num = get_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    callback_list = [tensorboard]

    # create model
    model = get_model()


    # start training
    print("----------start training---------")
    model.fit_generator(train_generator,
                        epochs=config.EPOCHS,
                        steps_per_epoch=train_num // config.BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // config.BATCH_SIZE,
                        callbacks=callback_list)

    # save the weights
    model.save_weights(filepath=config.save_model_dir, save_format='tf')
