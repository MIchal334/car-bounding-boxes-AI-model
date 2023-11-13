import os
import keras
from draw_squer import draw_squer
from keras.models import load_model
import cv2
from keras.models import Sequential
import matplotlib.pyplot as plt
from metrics import CustomIoUMetric
from metrics import custom_loss
import numpy as np

path_to_test_dictionary = '/home/michal/Desktop/photos/car_boxes/test_set'


def show_test_set(network):
    test_set = __load_images()
    for img in test_set:
        image_prepared = __preapre_image_for_proccesing(img)
        boxes = network.predict(image_prepared)
        draw_squer(boxes[0],image_prepared)


def __preapre_image_for_proccesing(img):
    x_size = 400
    y_size = 400
    image = cv2.resize(img, (x_size, y_size))
    image = image[np.newaxis, ...]
    return image.astype('float32')/255


def __load_images():
    images = []
    for filename in os.listdir(path_to_test_dictionary):
        img_path = os.path.join(path_to_test_dictionary, filename)
        images.append(cv2.imread(img_path, 1))
    return images 



if __name__ == "__main__":
    custom_objects = {'CustomIoUMetric': CustomIoUMetric, "custom_loss":custom_loss}
    network = keras.models.load_model('car_boxes_model',custom_objects=custom_objects)
    show_test_set(network)