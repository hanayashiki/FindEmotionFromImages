import numpy as np
import tensorflow as tf
from config import *

import keras.backend as K

def precision_k(y_true, y_pred, k):
    #one_hot_pred = K.zeros(shape=y_true.shape)
    #print(one_hot_pred.shape)
    # return K.cast(K.equal(K.argmax(y_true, axis=-1),
    #                       K.argmax(y_pred, axis=-1)),
    #               K.floatx())
    return \
        K.sum(
            K.cast(
                    K.cast(K.equal(K.argmax(y_true, axis=-1), k), K.floatx()) * #判断是0类
                    K.cast(K.equal(K.argmax(y_pred, axis=-1), k), K.floatx()) #实际是0类
            , K.floatx())
        ) / \
        K.sum(
            K.cast(K.equal(K.argmax(y_true, axis=-1), k), K.floatx())
        )

def precision_0_3(y_true, y_pred):
    return (precision_k(y_true, y_pred, 0) + precision_k(y_true, y_pred, 3)) / 2

def precision(y_true, y_pred):
    return (precision_k(y_true, y_pred, 0) + precision_k(y_true, y_pred, 1)
            + precision_k(y_true, y_pred, 2) + precision_k(y_true, y_pred, 3)) / 4

def precision_0(y_true, y_pred):
    return precision_k(y_true, y_pred, 0)

def precision_1(y_true, y_pred):
    return precision_k(y_true, y_pred, 1)

def precision_2(y_true, y_pred):
    return precision_k(y_true, y_pred, 2)

def precision_3(y_true, y_pred):
    return precision_k(y_true, y_pred, 3)