"""
helper functions
"""

import os
import numpy as np


def get_path(*dirs):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(base_path, *dirs)


data_path = get_path('data')

# ------------------------------------------------- #
# loading


def load_cities():
    """
    load a list of 30 cities
    """
    file = os.path.join(data_path, 'cities.txt')
    with open(file, 'r') as f:
        lines = f.readlines()
    return [line.split(',')[0] for line in lines]


def load_city_coordinates():
    """
    load a dictionary of cities and coordinates
    """
    file = os.path.join(data_path, 'cities.txt')
    city_coordinates = {}
    with open(file, 'r') as f:
        for line in f.readlines():
            city, lon, lat = line.strip().split(',')
            city_coordinates[city] = [float(lon), float(lat)]
    return city_coordinates


def load_category_topic_mapping():
    """
    load a dictionary of category, topic mapping
    """
    file = os.path.join(data_path, 'category_topics.txt')
    mapping = dict()
    with open(file, 'r') as f:
        for line in f.readlines():
            k, v = line.strip().split()
            mapping[int(k)] = v
    return mapping


def load_topics(topic_num=20):
    """
    load a list of topics
    """
    file = os.path.join(data_path, 'topics_mapping.txt')
    with open(file, 'r') as f:
        lines = f.readlines()
    return [line.split(',')[0] for line in lines]


def load_glove(embedding_dim=50):
    """
    build index mapping words in the embeddings set
    to their embedding vector (from GloVe)
    """
    print("Indexing GloVe word vectors...")
    embedding = {}
    file = os.path.join(data_path, 'glove',
                        f"glove.6B.{embedding_dim}d.txt")
    with open(file, "rb") as f:
        for line in f.readlines():
            values = line.decode().split()
            word = values[0]
            vec = np.asarray(values[1:], dtype=np.float)
            embedding[word] = vec
    return embedding

# ------------------------------------------------- #
