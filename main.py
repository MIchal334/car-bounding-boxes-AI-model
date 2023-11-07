from load_data import load_car_data
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.python.keras import backend
from check_history import show_all_history
from metrics import CustomIoUMetric
from metrics import custom_loss
from tensorflow.keras.layers import Dropout  


def __prepare_data(): 
    (train_images, valid_images, train_boxes, valid_boxes) = load_car_data()
    train_images = train_images.astype('float32')/255
    valid_images = valid_images.astype('float32')/255
    train_boxes = train_boxes.astype('float32')
    valid_boxes = valid_boxes.astype('float32')
    return  (train_images, train_boxes), (valid_images, valid_boxes)


def __get_model(shape):
    custom_iou = CustomIoUMetric()
    model = models.Sequential()
    model.add(layers.Conv2D(128, (12,12), activation = keras.activations.relu, input_shape = shape))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Conv2D(64, (6, 6), activation = keras.activations.relu))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(32, (3, 3), activation = keras.activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation = keras.activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation=keras.activations.relu))
    model.add(Dropout(0.5))
    model.add(layers.Dense(16, activation=keras.activations.selu))
    model.add(Dropout(0.5))
    model.add(layers.Dense(4, activation='linear'))
    model.compile(optimizer= keras.optimizers.Adam(), loss = custom_loss, metrics = [CustomIoUMetric()])
    model.summary()
    print('MODEL PREPERED')
    return model

def __divid_data(train_data: list,train_labels:list,max_size,amout_of_part):
    data_patrs = {}
    label_parts = {}

    for i in range(amout_of_part):
        data_patrs[i] = train_data[i*max_size:(i+1)*max_size]
        label_parts[i] = train_labels[i*max_size:(i+1)*max_size]

    yield data_patrs,label_parts


def train_network(train_data,train_labels,test_data,test_labels):
    max_size = 500
    amout_of_part = len(train_data) % max_size
    (data_patrs,label_parts) = __divid_data(train_data,train_labels,max_size,amout_of_part)
    network = __get_model(train_data[0].shape)
    for i in range(amout_of_part):
        history = network.fit(data_patrs[i],label_parts[i],batch_size = 4, epochs=2)


    return network.evaluate(test_data,test_labels, verbose=1)

if __name__ == "__main__":
    (train_images, train_boxes), (valid_images, valid_boxes) = __prepare_data()
    history = train_network(train_images,train_boxes,valid_images,valid_boxes)
    print(history)
    # show_all_history(history)