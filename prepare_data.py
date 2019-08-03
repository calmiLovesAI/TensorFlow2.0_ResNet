import tensorflow as tf
import config
import pathlib
from config import image_height, image_width, channels, proportion_of_test_set


def load_and_preprocess_image(img_path):
    # read pictures
    img_raw = tf.io.read_file(img_path)
    # decode pictures
    img_tensor = tf.image.decode_jpeg(img_raw, channels=channels)
    # resize
    img_tensor = tf.image.resize(img_tensor, [image_height, image_width])
    img_tensor = tf.cast(img_tensor, tf.float32)
    # normalization
    img = img_tensor / 255.0
    return img


def get_datasets():
    # get all images' paths (format: string)
    data_dir = config.dataset_dir
    data_root = pathlib.Path(data_dir)
    all_image_path = list(data_root.glob('*/*'))
    all_image_path = [str(path) for path in all_image_path]
    # get labels' names
    label_names = sorted(item.name for item in data_root.glob('*/'))
    # dict: name->index
    label_to_index = dict((index, name) for name, index in enumerate(label_names))
    # get all images' labels
    all_image_label = [label_to_index[pathlib.Path(p).parent.name] for p in all_image_path]

    # load dataset and preprocess images
    image_dataset = tf.data.Dataset.from_tensor_slices(all_image_path).map(load_and_preprocess_image)
    label_dataset = tf.data.Dataset.from_tensor_slices(all_image_label)
    dataset = tf.data.Dataset.zip((image_dataset, label_dataset))

    # spit the dataset into train and test
    image_count = len(all_image_path)
    test_count = int(image_count * proportion_of_test_set)
    train_count = image_count - test_count
    train_dataset = dataset.skip(test_count)
    test_dataset = dataset.take(test_count)

    # read the dataset in the form of batch
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE).repeat()
    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE).repeat()

    return train_dataset, test_dataset, train_count, test_count