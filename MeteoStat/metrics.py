from keras import metrics
import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def mse(img_1, img_2):
    return np.mean(tf.keras.metrics.mean_squared_error(img_1, img_2))






if __name__ == '__main__' :

    img_1  = mpimg.imread("radar2018_01_31_0745.png")
    img_2  = mpimg.imread("radar2018_01_31_1615.png")

    print(mse(img_1, img_2))
