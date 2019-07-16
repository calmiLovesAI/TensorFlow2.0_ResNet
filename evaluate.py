import tensorflow as tf
import config
from prepare_data import get_datasets

# get the dataset
train_dataset, test_dataset, train_count, test_count = get_datasets()

# load the model
new_model = tf.keras.models.load_model(config.model_dir)
# Get the accuracy on the test set
loss, acc = new_model.evaluate(test_dataset, steps=test_count // config.BATCH_SIZE)
print("The accuracy on test set is: {:6.3f}%".format(acc*100))