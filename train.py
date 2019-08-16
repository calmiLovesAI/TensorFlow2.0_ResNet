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
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  metrics=['accuracy'])

    model.fit(train_dataset,
              epochs=config.EPOCHS,
              steps_per_epoch=train_count // config.BATCH_SIZE,
              validation_data=valid_dataset,
              validation_steps=valid_count // config.BATCH_SIZE
              )



    # Use a custom training strategy
    # # optimizer
    # optimizer = tf.keras.optimizers.Adadelta()
    # # metrics
    # metric_train = tf.keras.metrics.Accuracy()
    # metric_valid = tf.keras.metrics.Accuracy()
    #
    # # start training
    # for epoch in range(config.EPOCHS):
    #     metric_train.reset_states()
    #     # train_total_correct = 0
    #     # train_image_num = 0
    #     for step, (image, label) in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             train_logits = model(image)
    #             label_one_hot = tf.one_hot(label, depth=config.NUM_CLASSES)
    #             loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y_true=label_one_hot,
    #                                                                            y_pred=train_logits,
    #                                                                            from_logits=True))
    #             train_prob = tf.nn.softmax(logits=train_logits, axis=1)
    #             train_pred = tf.argmax(train_prob, axis=1)
    #             metric_train.update_state(label, train_pred)
    #
    #         grads = tape.gradient(loss, model.trainable_variables)
    #         grads_and_vars = zip(grads, model.trainable_variables)
    #         optimizer.apply_gradients(grads_and_vars=grads_and_vars)
    #
    #         print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, train accuracy: {:.5f}".format(epoch + 1,
    #                                                                                        config.EPOCHS,
    #                                                                                        step + 1,
    #                                                                                        train_count // config.BATCH_SIZE,
    #                                                                                        loss,
    #                                                                                        metric_train.result().numpy()))
    #
    #     metric_valid.reset_states()
    #     for image, label in valid_dataset:
    #         logits = model(image)
    #         prob = tf.nn.softmax(logits=logits, axis=1)
    #         pred = tf.argmax(prob, axis=1)
    #         metric_valid.update_state(label, pred)
    #
    #     print("Epoch: {}/{}, valid accuracy: {:.5f}".format(epoch + 1, config.EPOCHS, metric_valid.result().numpy()))


    # save the weights
    model.save_weights(filepath=config.save_model_dir, save_format='tf')
