from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models import resnet50, resnet101, resnet152, resnet34
import config
from prepare_data import generate_datasets


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
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    # Use command tensorboard --logdir "log" to start tensorboard
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
    # callback_list = [tensorboard]

    # create model
    model = get_model()

    optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)

    # start training
    for epoch in range(config.EPOCHS):
        train_total_correct = 0
        train_image_num = 0
        for step, (image, label) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                train_logits = model(image)
                label_one_hot = tf.one_hot(label, depth=config.NUM_CLASSES)
                loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true=label_one_hot,
                                                                         y_pred=train_logits,
                                                                         from_logits=True))
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # calculate train accuracy
            train_prob = tf.nn.softmax(logits=train_logits, axis=1)
            train_pred = tf.cast(tf.argmax(train_prob, axis=1), dtype=tf.int32)
            correct_num = tf.reduce_sum(tf.cast(tf.equal(train_pred, label), dtype=tf.int32))
            train_total_correct += correct_num
            train_image_num += image.shape[0]
            train_accuracy = train_total_correct / train_image_num

            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, train accuracy: {:.5f}".format(epoch + 1,
                                                                                           config.EPOCHS,
                                                                                           step + 1,
                                                                                           train_count // config.BATCH_SIZE,
                                                                                           loss,
                                                                                           train_accuracy))

        total_correct = 0
        total_sum = 0
        for image, label in valid_dataset:
            logits = model(image)
            prob = tf.nn.softmax(logits=logits, axis=1)
            pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
            correct = tf.reduce_sum(tf.cast(tf.equal(pred, label), dtype=tf.int32))
            total_correct += int(correct)
            total_sum += image.shape[0]

        accuracy = total_correct / total_sum

        print("Epoch: {}/{}, valid accuracy: {:.5f}".format(epoch + 1, config.EPOCHS, accuracy))


    # save the weights
    model.save_weights(filepath=config.save_model_dir, save_format='tf')
