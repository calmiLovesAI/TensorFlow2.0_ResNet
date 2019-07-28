from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models import resnet50, resnet101, resnet152, resnet34
import config
from prepare_data import get_datasets

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


# get the dataset
train_dataset, test_dataset, train_count, test_count = get_datasets()

# Use command tensorboard --logdir 'log' to start tensorboard
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='log')
callback_list = [tensorboard]

# start training
# backbone = resnet50.ResNet50()
# if config.model == "resnet34":
backbone = resnet34.ResNet34()
# if config.model == "resnet101":
#     backbone = resnet101.ResNet101()
# if config.model == "resnet152":
#     backbone = resnet152.ResNet152()
model = tf.keras.Sequential()
model.add(backbone)
model.build(input_shape=(None, config.image_height, config.image_width, config.channels))

# model.compile(loss=tf.keras.losses.categorical_crossentropy,
#               optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
#               metrics=['accuracy'])
# model.summary()
#
# model.fit(train_dataset,
#           epochs=config.EPOCHS,
#           steps_per_epoch=train_count // config.BATCH_SIZE,
#           validation_data=test_dataset,
#           validation_steps=test_count // config.BATCH_SIZE,
#           callbacks=callback_list)




# optimizer = tf.keras.optimizers.SGD(learning_rate=config.learning_rate)
# for epoch in range(config.EPOCHS):
#     for step, (image, label) in enumerate(train_dataset):
#         with tf.GradientTape() as tape:
#             logits = model(image)
#             label_onehot = tf.one_hot(label, depth=config.NUM_CLASSES)
#             loss = tf.keras.losses.categorical_crossentropy(label_onehot, logits, from_logits=True)
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#         if step % 50 == 0:
#             print("epoch : %d, step : %d, loss : %d"%(epoch, step, loss))
#
#     total_num = 0
#     total_correct = 0
#     for image, label in test_dataset:
#         logits = model(image)
#         prob = tf.nn.softmax(logits, axis=1)
#         pred = tf.cast(tf.argmax(prob, axis=1), dtype=tf.int32)
#         # pred = tf.cast(pred, dtype=tf.int32)
#
#         correct = tf.reduce_sum(tf.cast(tf.equal(pred, label), dtype=tf.int32))
#
#         total_num += image.shape[0]
#         total_correct += int(correct)
#
#     accuracy = total_correct / total_num
#     print("Accuracy : %.3f"%(accuracy))



# save the whole model
model.save(config.model_dir)

