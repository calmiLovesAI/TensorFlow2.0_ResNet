# TensorFlow2.0_ResNet
A ResNet(**ResNet18, ResNet34, ResNet50, ResNet101, ResNet152**) implementation using TensorFlow-2.0

See https://github.com/calmisential/Basic_CNNs_TensorFlow2.0 for more CNNs.

## Train
1. Requirements:
+ Python >= 3.6
+ Tensorflow == 2.0.0
2. To train the ResNet on your own dataset, you can put the dataset under the folder **original dataset**, and the directory should look like this:
```
|——original dataset
   |——class_name_0
   |——class_name_1
   |——class_name_2
   |——class_name_3
```
3. Run the script **split_dataset.py** to split the raw dataset into train set, valid set and test set.
4. Change the corresponding parameters in **config.py**.
5. Run **train.py** to start training.
## Evaluate
Run **evaluate.py** to evaluate the model's performance on the test dataset.

## The networks I have implemented with tensorflow2.0:
+ [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152](https://github.com/calmisential/TensorFlow2.0_ResNet)
+ [InceptionV3](https://github.com/calmisential/TensorFlow2.0_InceptionV3)


## References
1. The original paper: https://arxiv.org/abs/1512.03385
2. The TensorFlow official tutorials: https://tensorflow.google.cn/beta/tutorials/quickstart/advanced