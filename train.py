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

    return model


if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # get the original_dataset
    train_dataset, test_dataset, train_count, test_count = get_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    # callback_list = [tensorboard]

    # create model
    model = get_model()

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    # start training
    for epoch in range(config.EPOCHS):
        for step, (image, label) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(image)
                label_one_hot = tf.one_hot(label, depth=config.NUM_CLASSES)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true=label_one_hot,
                                                                         y_pred=logits,
                                                                         from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}".format(epoch,
                                                                   config.EPOCHS,
                                                                   step,
                                                                   train_count // config.BATCH_SIZE,
                                                                   loss))

        total_correct = 0
        total_sum = 0
        for image, label in test_dataset:
            logits = model(image)
            prob = tf.nn.softmax(logits=logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, label), dtype=tf.int32))
            total_correct += int(correct)
            total_sum += image.shape[0]

        accuracy = total_correct / total_sum

        print("Epoch: {}/{}, accuracy: {:.5f}".format(epoch, config.EPOCHS, accuracy))


    # save the weights
    model.save_weights(filepath=config.save_model_dir, save_format='tf')
