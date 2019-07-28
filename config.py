# some training parameters
EPOCHS = 10
BATCH_SIZE = 8
NUM_CLASSES = 2
image_height = 224
image_width = 224
channels = 3
model_dir = "resnet_model.h5"
dataset_dir = "dataset/"
proportion_of_test_set = 0.2
learning_rate = 0.001

# choose a network
model = "resnet34"
# model = "resnet50"
# model = "resnet101"
# model = "resnet152"
