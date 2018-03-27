from __future__ import division, unicode_literals, print_function
from sacred import Experiment
from Model import create_model
from TrainingDataGenerator import RandomlyEditedVideoTrainingDataGenerator
from keras.optimizers import Adam
from os import path

ex = Experiment("CutFinder")


@ex.config
def config():
    bs = 1
    lr = 0.001
    lr_decay = 0.0
    image_size = (16*16, 9*16)
    filter_counts = [64, 128, 256, 256, 256, 256]
    strides = [1, 2, 2, 1, 2, 2]


@ex.automain
def train(bs, lr, lr_decay, image_size, filter_counts, strides):
    optimizer = Adam(lr, decay=lr_decay)
    model = create_model(image_size, filter_counts, strides)
    model.compile(optimizer, loss="mse", metrics=["accuracy"])
    data_generator = RandomlyEditedVideoTrainingDataGenerator(
        path.join("..", "..", "..", "Datasets", "VideoClips"),
        batch_size=bs, image_size=image_size)
    for x, y in data_generator:
        model.fit(x, y)
