"""
helper functions
"""

import os

#-------------------------------------------------#

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
data_path = os.path.join(base_path, 'data')
results_path = os.path.join(base_path, 'results')

def load_cities():
    """
    load a list of 30 cities
    """
    CITY_FILE = 'cities.txt'
    file = os.path.join(data_path, CITY_FILE)
    with open(file, 'r') as f:
        lines = f.readlines()
    return [line.split(',')[0] for line in lines]


def load_city_coordinates():
    """
    load a dictionary of cities and coordinates
    """
    CITY_FILE = 'cities.txt'
    file = os.path.join(data_path, CITY_FILE)
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
    MAPPING_FILE = 'category_topics.txt'
    file = os.path.join(data_path, MAPPING_FILE)
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
    TOPIC_FILE = 'topics_mapping.txt'
    file = os.path.join(data_path, TOPIC_FILE)
    with open(file, 'r') as f:
        lines = f.readlines()
    return [line.split(',')[0] for line in lines]
#-------------------------------------------------#