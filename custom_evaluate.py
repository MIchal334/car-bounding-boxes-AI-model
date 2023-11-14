from metrics import CustomIoUMetric
from metrics import custom_loss
import numpy as np
import tensorflow as tf
from tensorflow import keras
import csv
import os
import cv2

#cordinate [x_min,y_min,x_max,y_max]
x_size = 416
y_size = 416
file_train_valid = '/home/michal/Desktop/photos/car_boxes/boxes/valid'
file_csv_name = '_annotations.csv'

def custom_evaluate():
    error_sum = 0
    pthot_sum = 0
    network = __get_model()
    with open(os.path.join(file_train_valid,file_csv_name)) as csvfile:
        reader = csv.DictReader(csvfile, quotechar='|')
        for row in reader:
            img_name = row['filename']
            img = __get_image(file_train_valid,img_name)
            if img is not None:
                pthot_sum = pthot_sum + 1 
                true_cordinte = [int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])]
                predict_cordinate = __get_predict_coordinate(img,network)
                error_sum = error_sum + __clacluate_error(true_cordinte,predict_cordinate)

    err = error_sum / pthot_sum
    print('ERROR: ',err)


def __get_model():
    custom_objects = {'CustomIoUMetric': CustomIoUMetric, "custom_loss":custom_loss}
    return keras.models.load_model('car_boxes_model',custom_objects=custom_objects)

def __clacluate_error(true_cordinte,predict_cordinate):
    true_area = (int(true_cordinte[2]) - int(true_cordinte[0])) * (int(true_cordinte[3]) - int(true_cordinte[1]))
    x_coverage = __dim_mesurment([true_cordinte[0],true_cordinte[2]],[predict_cordinate[0],predict_cordinate[2]])
    y_coverage = __dim_mesurment([true_cordinte[1],true_cordinte[3]],[predict_cordinate[1],predict_cordinate[3]]) 
    predic_area = x_coverage * y_coverage
    div_area =  predic_area/true_area
    return abs(1 - div_area)     
    
def __dim_mesurment(X_true,X_predict):
    if X_true[0] > X_predict[0] and X_true[1] > X_predict[1]:
       return X_predict[1] -  X_true[0]

    elif X_predict[0] > X_true[0] and X_predict[1] > X_true[1]:
        return X_true[1] - X_predict[0] 
    
    return X_predict[1] - X_predict[0]


def __get_image(path,image_name):
    img_path = os.path.join(path, image_name)
    if not os.path.exists(img_path):
        return None
    
    img_pix = cv2.imread(img_path, 1)
    return cv2.resize(img_pix, (x_size, y_size))


def __get_predict_coordinate(img, network):
    image_prepared = __preapre_image_for_proccesing(img)
    boxes = network.predict(image_prepared)
    for c in boxes[0]:
        c = int(c)
    return boxes[0]




def __preapre_image_for_proccesing(img):
    x_size = 416
    y_size = 416
    image = cv2.resize(img, (x_size, y_size))
    image = image[np.newaxis, ...]
    return image.astype('float32')/255

if __name__ == "__main__":
    custom_evaluate()