#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, TimeDistributed, Permute, Reshape, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


def create_model(image_size=(1280, 720), filter_counts=[64, 128, 256, 256, 256, 256], strides=[1, 2, 2, 1, 2, 2]):
    model = Sequential()
    model.add(TimeDistributed(Conv2D(filter_counts[0], kernel_size=5, strides=strides[0],
                                     kernel_initializer=RandomNormal(0, 0.02),
                                     use_bias=True, padding="same", data_format="channels_last",
                                     input_shape=(image_size[0], image_size[1], 3)),
                              input_shape=(3, image_size[0], image_size[1], 3), name="TimeDistributedConvLayer1"))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Conv2D(filter_counts[1], kernel_size=5, strides=strides[1],
                                     kernel_initializer=RandomNormal(0, 0.02),
                                     use_bias=False, padding="same", data_format="channels_last")))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(TimeDistributed(Conv2D(filter_counts[2], kernel_size=5, strides=strides[2],
                                     kernel_initializer=RandomNormal(0, 0.02),
                                     use_bias=False, padding="same", data_format="channels_last")))
    model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
    model.add(Permute((2, 3, 1, 4), name="PermuteLayer"))
    resred = strides[0] * strides[1] * strides[2]
    model.add(Reshape((image_size[0]//resred, image_size[1]//resred, 3*filter_counts[2]),
                      name="ReshapeAfterPermuteLayer"))
    model.add(Conv2D(filter_counts[3], kernel_size=3, strides=strides[3],
                     kernel_initializer=RandomNormal(0, 0.02),
                     use_bias=True, padding="same", data_format="channels_last", name="JoinedConvLayer1"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(filter_counts[4], kernel_size=4, strides=strides[4],
                     kernel_initializer=RandomNormal(0, 0.02),
                     use_bias=False, padding="same", data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(filter_counts[5], kernel_size=4, strides=strides[5],
                     kernel_initializer=RandomNormal(0, 0.02),
                     use_bias=False, padding="same", data_format="channels_last"))
    model.add(LeakyReLU(alpha=0.2))
    resred = resred * strides[3] * strides[4] * strides[5]
    model.add(Reshape((image_size[0]//resred*image_size[1]//resred*filter_counts[5],),
                      name="ReshapeForDenseOutput"))
    model.add(Dense(2, activation="softmax", use_bias=True, name="Output"))
    return model
