#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Conv2D, TimeDistributed, Permute, Reshape, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


model = Sequential()
model.add(TimeDistributed(Conv2D(64, kernel_size=5, strides=1, kernel_initializer=RandomNormal(0, 0.02),
                                 use_bias=True, padding="same", data_format="channels_last",
                                 input_shape=(1280, 720, 3)),
                          input_shape=(3, 1280, 720, 3), name="TimeDistributedConvLayer1"))
model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
model.add(TimeDistributed(Conv2D(128, kernel_size=5, strides=2, kernel_initializer=RandomNormal(0, 0.02),
                                 use_bias=False, padding="same", data_format="channels_last")))
model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
model.add(TimeDistributed(Conv2D(256, kernel_size=5, strides=2, kernel_initializer=RandomNormal(0, 0.02),
                                 use_bias=False, padding="same", data_format="channels_last")))
model.add(TimeDistributed(LeakyReLU(alpha=0.2)))
model.add(Permute((2, 3, 1, 4), name="PermuteLayer"))
model.add(Reshape((1280//4, 720//4, 3*256), name="ReshapeAfterPermuteLayer"))
model.add(Conv2D(256, kernel_size=3, strides=1, kernel_initializer=RandomNormal(0, 0.02),
                 use_bias=True, padding="same", data_format="channels_last", name="JoinedConvLayer1"))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(256, kernel_size=4, strides=2, kernel_initializer=RandomNormal(0, 0.02),
                 use_bias=False, padding="same", data_format="channels_last"))
model.add(LeakyReLU(alpha=0.2))
model.add(Conv2D(256, kernel_size=4, strides=2, kernel_initializer=RandomNormal(0, 0.02),
                 use_bias=False, padding="same", data_format="channels_last"))
model.add(LeakyReLU(alpha=0.2))
model.add(Reshape((1280//16*720//16*256,), name="ReshapeForDenseOutput"))
model.add(Dense(2, activation="softmax", use_bias=True, name="Output"))
