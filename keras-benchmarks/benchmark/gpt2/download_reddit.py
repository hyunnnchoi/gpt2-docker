import keras
import keras_nlp

import benchmark
from benchmark import utils

import os
import json

os.environ["KERAS_BACKEND"] = "tensorflow"

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import time

if __name__ == "__main__":
    reddit_ds = tfds.load("reddit", split="train", as_supervised=True, data_dir='/data/reddit')
