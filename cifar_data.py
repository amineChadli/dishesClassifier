# cifar data

import tensorflow as tf
import numpy as np
import pickle

DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_FOLDER_NAME = "cifar-10-batches-py"
# Various constants for the size of the images.
# Use these constants in your own program.

# Width and height of each image.
img_size = 32

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 10
########################################################################
# Various constants used to allocate arrays of the correct size.

# Number of files for the training-set.
_num_files_train = 5

# Number of images for each batch-file in the training-set.
_images_per_file = 10000

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = _num_files_train * _images_per_file

########################################################################
def _unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def _maybe_download():

    return "./cifar-10-batches-py"
    path = tf.keras.utils.get_file(DATA_URL.split('/')[-1], DATA_URL, cache_subdir='datasets\\cifar' , extract = True) 
    path_split = path.split("\\")
    
    path_split.pop(-1)
    path_split.append(DATA_FOLDER_NAME)
    path = "\\".join(path_split) 
    return path

def _load_class_names():
    """
    Load the names for the classes in the CIFAR-10 data-set.
    Returns a list with the names. Example: names[3] is the name
    associated with class-number 3.
    """

    # Load the class-names from the pickled file.
    path = _maybe_download()+"/batches.meta"
    raw = _unpickle(path)[b'label_names']

    # Convert from binary strings.
    names = [x.decode('utf-8') for x in raw]

    return names

def _convert_images(raw):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    """

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=np.float32) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images
def _load_data(filename):
    dict = _unpickle(filename)
    # Get the raw images.
    raw_images = dict[b'data']
    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(dict[b'labels'])
    # Convert the images.
    images = _convert_images(raw_images)
    return images , cls

def _load_training_data():
    path = _maybe_download()+"/data_batch_"
    # Pre-allocate the arrays for the images and class-numbers for efficiency.
    images = np.zeros(shape=[_num_images_train, img_size, img_size, num_channels], dtype=np.float32)
    cls = np.zeros(shape=[_num_images_train], dtype=int)

    # Begin-index for the current batch.
    begin = 0

    # For each data-file.
    for i in range(_num_files_train):
        # Load the images and class-numbers from the data-file.
        filename = path + str(i + 1)
        images_batch, cls_batch = _load_data(filename)

        # Number of images in this batch.
        num_images = len(images_batch)

        # End-index for the current batch.
        end = begin + num_images

        # Store the images into the array.
        images[begin:end, :] = images_batch

        # Store the class-numbers into the array.
        cls[begin:end] = cls_batch

        # The begin-index for the next batch is the current end-index.
        begin = end

    return images, cls 

def _load_test_data():
    path = _maybe_download()+"/test_batch"
    return _load_data(path)

def _load_all_data():
    _load_training_data()
    _load_test_data()
    _load_class_names()
_load_all_data()