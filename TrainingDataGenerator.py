#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function
from keras.utils import Sequence
from moviepy.editor import VideoFileClip
from os import path, listdir
from random import choice, randrange, random
import re
import numpy as np


class NoUsableTrainingData(Exception):
    pass


class RandomlyEditedVideoTrainingDataGenerator(Sequence):
    def __init__(self, video_folder):
        self.video_folder = video_folder
        self.current_clip = None
        self.current_clip_filename = None
        self.filename_regex = re.compile(r".*\.(mp4|webm|avi|ogv)$")
        self.last_frames = []
        self.current_pos = 0

    def __len__(self):
        return 10

    @staticmethod
    def get_clip_frame_count(clip):
        return int(np.floor(clip.duration * clip.fps))

    def ensure_clip(self, file_list):
        if len(file_list) <= 0:
            raise NoUsableTrainingData()
        fail_counter = 0
        while not self.current_clip:
            try:
                self.current_clip_filename = choice(file_list)
                self.current_clip = VideoFileClip(self.current_clip_filename)
                new_start_frame = randrange(0, self.get_clip_frame_count(self.current_clip) - 2)
                self.current_pos = new_start_frame*1/self.current_clip.fps
            except IndexError or OSError:
                fail_counter += 1
            if fail_counter > 10:
                raise NoUsableTrainingData()

    def next(self, file_list):
        self.ensure_clip(file_list)
        while len(self.last_frames) < 2:
            self.last_frames.append(self.current_clip.get_frame(self.current_pos))
            self.current_pos += 1 / self.current_clip.fps
        # 20% Chance for a cut
        if random() < 0.2 or self.current_pos + 2 / self.current_clip.fps > self.current_clip.duration:
            self.current_clip = None
            filtered_file_list = list(file_list)
            filtered_file_list.remove(self.current_clip_filename)
            self.ensure_clip(filtered_file_list)
            self.last_frames = self.last_frames[-2:]
            self.last_frames.append(self.current_clip.get_frame(self.current_pos))
            data_set = np.array(self.last_frames), np.array([1.0, 0.0])
            self.current_pos += 1 / self.current_clip.fps
            self.last_frames.append(self.current_clip.get_frame(self.current_pos))
            return data_set
        else:
            self.last_frames = self.last_frames[-2:]
            self.current_pos += 1 / self.current_clip.fps
            self.last_frames.append(self.current_clip.get_frame(self.current_pos))
            return np.array(self.last_frames), np.array([0.0, 1.0])

    def __getitem__(self, idx):
        file_list = listdir(self.video_folder)
        for i, filename in enumerate(file_list):
            if not self.filename_regex.match(filename):
                file_list.pop(i)
        batch_X = []
        batch_y = []
        for i in range(self.__len__()):
            x, y = self.next([path.join(self.video_folder, filename) for filename in file_list])
            batch_X.append(x)
            batch_y.append(y)
        return np.array(batch_X), np.array(batch_y)


if __name__ == "__main__":
    generator = RandomlyEditedVideoTrainingDataGenerator(path.join("..", "training_data"))
    print(generator.__getitem__(0))
