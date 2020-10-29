import os

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class ResBlock(layers.Layer):
    
    def __init__(self):
        self.conv1 = layers.Conv2D(256,kernel_size=4, strides=2,padding='same', use_bias=False)
        self.batchnorm = layers.BatchNormalisation()
        self.relu = layers.ReLU()
        
    def call(self, x):
        return x + self.relu(self.batchnorm(self.conv1(x)))
