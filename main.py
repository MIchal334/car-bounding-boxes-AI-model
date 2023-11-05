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
from tensorflow.keras.metrics import Metric


class CustomIoUMetric(Metric):
    def __init__(self, name='iou', **kwargs):
        super(CustomIoUMetric, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight('intersection', initializer='zeros')
        self.union = self.add_weight('union', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
        union = tf.reduce_sum(tf.maximum(y_true, y_pred))
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        iou = self.intersection / (self.union + tf.keras.backend.epsilon())
        return iou

def __iou_loss(y_true, y_pred):
    intersection = tf.reduce_sum(tf.minimum(y_true, y_pred))
    union = tf.reduce_sum(tf.maximum(y_true, y_pred))
    iou = intersection / (union + tf.keras.backend.epsilon())
    return 1 - iou

@tf.function
def __custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return __iou_loss(y_true, y_pred)

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
    model.add(layers.Conv2D(128, (3,3), activation = keras.activations.relu, input_shape = shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation = keras.activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation = keras.activations.relu))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64, activation=keras.activations.relu))
    model.add(layers.Dense(16, activation=keras.activations.selu))
    model.add(layers.Dense(4, activation='linear'))
    model.compile(optimizer= keras.optimizers.Adam(), loss = __custom_loss, metrics = [custom_iou])
    model.summary()
    print('MODEL PREPERED')
    return model

def train_network(train_data,train_labels,test_data,test_labels):
    network = __get_model(train_data[0].shape)
    return network.fit(train_data,train_labels,batch_size = 2, validation_data=(test_data,test_labels), epochs=50)


if __name__ == "__main__":
    (train_images, train_boxes), (valid_images, valid_boxes) = __prepare_data()
    history = train_network(train_images,train_boxes,valid_images,valid_boxes)
    show_all_history(history)