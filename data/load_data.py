# cifar data

import tensorflow as tf
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Width and height of each image.
img_size = 224

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 3
########################################################################

# This is used to pre-allocate arrays for efficiency.
#_num_images_train = _num_files_train * _images_per_file

########################################################################
def _load_class_names():

    names = ["bastilla","couscous","tagine"]

    return names

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    """
    
    raw/=255.0
    # Convert the raw images from the data-files to floating-points.
    

    # Reshape the array to 4-dimensions.
    images = raw.reshape([-1,img_size, img_size,num_channels])

    return images

def _convert_cls(cls):
    
    labelencoder = LabelEncoder()
    cls = labelencoder.fit_transform(cls)
    return cls
    

def _load_training_data(training_data_path,training_labels_path):
    # Get the raw images.
    raw_images = np.load(training_data_path)
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.load(training_labels_path)
    # Convert the images.
    images = _convert_images(raw_images)
    cls = _convert_cls(cls)
    
    return images , cls

def _load_test_data(test_data_path,test_labels_path):
    # Get the raw images.
    raw_images = np.load(test_data_path)
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.load(test_labels_path)
    # Convert the images.
    images = _convert_images(raw_images)
    cls = _convert_cls(cls)

    return images , cls
