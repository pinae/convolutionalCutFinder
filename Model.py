#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, Conv2D, TimeDistributed, Permute, Reshape, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


inp = Input(batch_shape=(None,), shape=(1280, 720, 3))
x = TimeDistributed(Conv2D(64, kernel_size=5, strides=1, kernel_initializer=RandomNormal(0, 0.02), use_bias=True,
                           padding="same", data_format="channels_last")(inp))
x = TimeDistributed(LeakyReLU(alpha=0.2)(x))
x = TimeDistributed(Conv2D(128, kernel_size=5, strides=2, kernel_initializer=RandomNormal(0, 0.02), use_bias=False,
                           padding="same", data_format="channels_last")(x))
x = TimeDistributed(LeakyReLU(alpha=0.2)(x))
x = TimeDistributed(Conv2D(256, kernel_size=5, strides=2, kernel_initializer=RandomNormal(0, 0.02), use_bias=False,
                           padding="same", data_format="channels_last")(x))
x = TimeDistributed(LeakyReLU(alpha=0.2)(x))
x = Reshape((1280, 720, 3*256))(Permute((2, 3, 1, 4))(x))
x = Conv2D(256, kernel_size=3, strides=1, kernel_initializer=RandomNormal(0, 0.02), use_bias=True,
           padding="same", data_format="channels_last")(x)
x = LeakyReLU(alpha=0.2)(x)
x = Conv2D(256, kernel_size=4, strides=2, kernel_initializer=RandomNormal(0, 0.02), use_bias=False,
           padding="same", data_format="channels_last")(x)
x = LeakyReLU(alpha=0.2)(x)
x = Conv2D(256, kernel_size=4, strides=2, kernel_initializer=RandomNormal(0, 0.02), use_bias=False,
           padding="same", data_format="channels_last")(x)
x = LeakyReLU(alpha=0.2)(x)
out = Dense(2, activation="softmax", use_bias=True)(x)
model = Model(inputs=[inp], outputs=out)
