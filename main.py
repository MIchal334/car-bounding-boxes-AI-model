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
    model = keras.Sequential()
    model.add(layers.Conv2D(64, (6, 6), activation='relu', input_shape=shape))
    model.add(layers.MaxPooling2D((4, 4)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))
    model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(16, activation='relu'))
    # model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='linear'))
    model.compile(optimizer= keras.optimizers.Adam(), loss = custom_loss, metrics = [CustomIoUMetric()])
    model.summary()
    print('MODEL PREPERED')
    return model




def train_network(train_data,train_labels,test_data,test_labels):
    network = __get_model(train_data[0].shape)
    history = network.fit(train_data,train_labels,batch_size = 2, validation_data=(test_data,test_labels), epochs=60)
    network.save('car_boxes_model')
    return history



if __name__ == "__main__":
    (train_images, train_boxes), (valid_images, valid_boxes) = __prepare_data()
    history = train_network(train_images,train_boxes,valid_images,valid_boxes)
    show_all_history(history)

