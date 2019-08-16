import tensorflow as tf
import config
from prepare_data import generate_datasets
from train import get_model

if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    # print(train_dataset)
    # load the model
    new_model = get_model()

    new_model.load_weights(filepath=config.save_model_dir)
    print("----------start evaluating---------")
    # Get the accuracy on the test set
    loss, acc = new_model.evaluate(test_dataset, steps=test_count // config.BATCH_SIZE)
    print("The accuracy on test set is: {:6.3f}%".format(acc*100))