from __future__ import division, unicode_literals, print_function
from sacred import Experiment
from Model import model
from TrainingDataGenerator import RandomlyEditedVideoTrainingDataGenerator
from keras.optimizers import Adam
from os import path

ex = Experiment("CutFinder")


@ex.config
def config():
    lr = 0.001
    lr_decay = 0.0


@ex.automain
def train(lr, lr_decay):
    optimizer = Adam(lr, decay=lr_decay)
    model.compile(optimizer, loss="mse", metrics=["accuracy"])
    data_generator = RandomlyEditedVideoTrainingDataGenerator(
        path.join("..", "..", "..", "Datasets", "VideoClips"))
    for x, y in data_generator:
        model.fit(x, y)
