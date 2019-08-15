# some training parameters
EPOCHS = 50
BATCH_SIZE = 16
NUM_CLASSES = 5
image_height = 224
image_width = 224
channels = 3
save_model_dir = "saved_model/"
dataset_dir = "dataset/"
train_dir = dataset_dir + "train/"
valid_dir = dataset_dir + "valid/"
test_dir = dataset_dir + "test/"
proportion_of_test_set = 0.2
learning_rate = 0.001

# choose a network
model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"
