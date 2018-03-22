#!/usr/bin/python3
# -*- coding: utf-8 -*-
from keras.models import Model
from keras.layers import Input, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal


def conv_block_d(input_tensor, f, use_instance_norm=True):
    x = input_tensor
    x = Conv2D(f, kernel_size=4, strides=2, kernel_initializer=RandomNormal(0, 0.02), use_bias=False, padding="same")(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


inp = Input(batch_shape=(None,), shape=(1280, 720, 3))
x = Conv2D(64, kernel_size=5, strides=1, kernel_initializer=RandomNormal(0, 0.02), use_bias=True,
           padding="same", data_format="channels_last")(inp)
x = LeakyReLU(alpha=0.2)(x)
#x = GaussianNoise(0.05)(inp)
x = conv_block_d(inp, 64, False)
x = conv_block_d(x, 128, False)
x = conv_block_d(x, 256, False)
out = Conv2D(1, kernel_size=4, kernel_initializer=RandomNormal(0, 0.02), use_bias=False, padding="same", activation="sigmoid")(x)
model = Model(inputs=[inp], outputs=out)