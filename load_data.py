import csv
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from draw_squer import draw_squer

file_train_data = '/home/michal/Desktop/photos/car_boxes/boxes/train'
file_train_valid = '/home/michal/Desktop/photos/car_boxes/boxes/valid'
file_csv_name = '_annotations.csv'
x_size = 416
y_size = 416

def load_car_data():
    train_images, train_boxes = __load_data_from_csv(file_train_data)
    print('TRAIN DATA LOADED')
    valid_images, valid_boxes = __load_data_from_csv(file_train_valid)
    print('VALID DATA LOADED')
    return (np.array(train_images), np.array(valid_images), np.array(train_boxes),np.array(valid_boxes))


def __load_data_from_csv(path_dir):
    images = []
    bounding_boxes = []
    with open(os.path.join(path_dir,file_csv_name)) as csvfile:
            reader = csv.DictReader(csvfile, quotechar='|')
            for row in reader:
                img_name = row['filename']
                image = __load_cv2_image(path_dir,img_name)
                bounding_box = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
                images.append(image)
                bounding_boxes.append(bounding_box)
    return (images, bounding_boxes)


def __load_cv2_image(path,img):
    img_path = os.path.join(path, img)
    img_pix = cv2.imread(img_path, 1)
    return cv2.resize(img_pix, (x_size, y_size))