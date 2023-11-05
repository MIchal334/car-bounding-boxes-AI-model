from tensorflow.keras.metrics import Metric
import tensorflow as tf

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
def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return __iou_loss(y_true, y_pred)