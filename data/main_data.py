# main_data

import load_data as data_loader
import data_convertor

training_data_path = "../dataset/training_set.npy"

training_labels_path = "../dataset/training_labels.npy"

list_images , list_classes = data_loader._load_training_data(training_data_path,training_labels_path)

dest = "../tf_records"
data_convertor.convert(list_images,list_classes, dest,shuffle=False)