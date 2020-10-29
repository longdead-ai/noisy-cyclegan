import os

import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential


class ResBlock(layers.Layer):
    
    def __init__(self, filters):
        self.conv = layers.Conv2D(filters,kernel_size=4, strides=2,padding='same', use_bias=False)
        self.batchnorm = layers.BatchNormalisation()
        self.relu = layers.ReLU()
        
    def call(self, x):
        return self.relu(self.batchnorm(self.conv1(x)))


class TransResBlock(layers.Layer):
    
    def __init__(self, filters):
        self.convtrans = layers.Conv2DTranspose(filters, kernel_size=4, strides=2, padding='same', use_bias=False)
        self.batchnorm = layers.BatchNormalisation()
        self.relu = layers.ReLU()
        
    def call(self, x):
        return self.relu(self.batchnorm(self.conv(x)))
