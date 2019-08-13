import tensorflow as tf
import config
# import pathlib
# from config import image_height, image_width, channels, proportion_of_test_set


def get_datasets():
    # Preprocess the original_dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )

    train_generator = train_datagen.flow_from_directory(config.train_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=1,
                                                        shuffle=True,
                                                        class_mode="categorical")

    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0
    )
    valid_generator = valid_datagen.flow_from_directory(config.valid_dir,
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=7,
                                                        shuffle=True,
                                                        class_mode="categorical"
                                                        )
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 /255.0
    )
    test_generator = test_datagen.flow_from_directory(config.test_dir,
                                                      target_size=(config.image_height, config.image_width),
                                                      color_mode="rgb",
                                                      batch_size=config.BATCH_SIZE,
                                                      seed=7,
                                                      shuffle=True,
                                                      class_mode="categorical"
                                                      )


    train_num = train_generator.samples
    valid_num = valid_generator.samples
    test_num = test_generator.samples


    return train_generator, \
           valid_generator, \
           test_generator, \
           train_num, valid_num, test_num

# def load_and_preprocess_image(img_path):
#     # read pictures
#     img_raw = tf.io.read_file(img_path)
#     # decode pictures
#     img_tensor = tf.image.decode_jpeg(img_raw, channels=channels)
#     # resize
#     img_tensor = tf.image.resize(img_tensor, [image_height, image_width])
#     img_tensor = tf.cast(img_tensor, tf.float32)
#     # normalization
#     img = img_tensor / 255.0
#     return img
#
#
# def get_datasets():
#     # get all images' paths (format: string)
#     data_dir = config.dataset_dir
#     data_root = pathlib.Path(data_dir)
#     all_image_path = list(data_root.glob('*/*'))
#     all_image_path = [str(path) for path in all_image_path]
#     # get labels' names
#     label_names = sorted(item.name for item in data_root.glob('*/'))
#     # dict: name->index
#     label_to_index = dict((index, name) for name, index in enumerate(label_names))
#     # get all images' labels
#     all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]
#
#     # load original_dataset and preprocess images
#     image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
#     label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
#     original_dataset = tf.data.Dataset.zip((image_dataset, label_dataset))
#
#     # spit the original_dataset into train and test
#     image_count = len(all_image_path)
#     test_count = int(image_count * proportion_of_test_set)
#     train_count = image_count - test_count
#     train_dataset = original_dataset.skip(test_count)
#     test_dataset = original_dataset.take(test_count)
#
#     # read the original_dataset in the form of batch
#     train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE).repeat()
#     test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE).repeat()
#
#     return train_dataset, test_dataset, train_count, test_count