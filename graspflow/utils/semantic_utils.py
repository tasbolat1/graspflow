#!/usr/bin/env python3
import json
import os
import numpy as np
import pickle as pickle

def load_glove_model(File):
    print("Loading Glove Model")
    glove_model = {}
    with open(File,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            glove_model[word] = embedding
    print(f"{len(glove_model)} words loaded!")
    return glove_model


def word2vector(word, model="glove"):

    # use the glove representation of the task
    if model == "glove":
        cache_filename = "data/glove_data/cache.pkl"
        if os.path.exists(cache_filename): # read data from cache to speed up
            with open(cache_filename, 'rb') as input:
                cache = pickle.load(input)
            vector = cache.get(word,0)
            # task is not cached
            if isinstance(vector,type(0)):
                glove_data = load_glove_model("data/glove_data/glove.6B.300d.txt")
                vector = glove_data[word]
                cache[word] = vector
            with open(cache_filename, 'wb') as fp:
                pickle.dump(cache, fp)
            return vector
        else:
            glove_data = load_glove_model("data/glove_data/glove.6B.300d.txt")
            vector = glove_data[word]
            cache = {}
            cache[word] = vector
            with open(cache_filename, 'wb') as fp:
                pickle.dump(cache, fp)
                print("data cached!")
            return vector


