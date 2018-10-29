#utils.py
import sys , os
import tensorflow as tf
import six
import load_data as data_loader
import data_wrapper as data_wrapper
import numpy as np
from matplotlib import pyplot as plt
import random
import cv2

DEBUG = False

# define a function to wrap data 
def wrap_data(writer,image,label):

    image_bytes = image.tostring()

    # Create a dict with the data we want to save in the
    # TFRecords file. You can add more relevant data here.
    data = \
            {
                'image': data_wrapper._wrap_bytes(image_bytes),
                'label' : data_wrapper._int64_feature(label)
            }

    # Wrap the data as TensorFlow Features.
    feature = tf.train.Features(feature=data)

    # Wrap again as a TensorFlow Example.
    example = tf.train.Example(features=feature)
    # Serialize the data.
    serialized = example.SerializeToString()
                    
    # Write the serialized data to the TFRecords file.
    writer.write(serialized)

def convert_data(writer , list_images, classes):
    # Number of images in the source that have at least one core point
    N = len(list_images)

    # total number of batches
    total_batches = N #(N + batch_size)//batch_size
    print("creating tf-record : \n- Total batchs : " , total_batches)

    for image,cls in zip(list_images,classes):
        wrap_data(writer,image,cls)

def convert(list_images,list_classes, dest,shuffle=False):
    # Args:
    # src     path of images
    # dest    File-path for the TFRecords output file.
    # batch_size : batch size for reading images
    flag = "Train"
    out_path_Train = dest + "/" + flag + ".tf" 
    
    
    # Open a TFRecordWriter for the output-file.
    # Create tf-record for training data
    
    print("\n-->Training data...")
    with tf.python_io.TFRecordWriter(out_path_Train) as train_writer:
        convert_data(train_writer , list_images,list_classes)
    

def convert_test_image(list_images, out_path):
    # Args:
    # image_paths   List of file-paths for the images.
    # labels        Class-labels for the images.
    # out_path      File-path for the TFRecords output file.
    
    print("\nConverting: " + out_path)
    

    
    # Open a TFRecordWriter for the output-file.
    with tf.python_io.TFRecordWriter(out_path) as writer:
        
        # Iterate over all the positif blocs and class-labels.
        for img in list_images:

            img_bytes = img.tostring()

            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
          
            data = \
                {
                    'image': data_wrapper._wrap_bytes(img_bytes)
                }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()
            
            # Write the serialized data to the TFRecords file.
            writer.write(serialized)
            
'''
src = "../../dataset/training_data"
dest = "../../dataset/tf-records/training"
convert(src,dest,4)            

src = "../../dataset/validation_data"
dest = "../../dataset/tf-records/validation"
convert(src,dest,4)'''
