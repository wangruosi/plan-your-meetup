"""
Set up PySpark data structure

Ruosi Wang ruosiwang.psy@gmail.com
"""
from .helper import load_category_topic_mapping

import os
import re
import gzip
import ujson as json
from itertools import islice

from datetime import datetime
from pytz import timezone
from pyspark.sql import SQLContext, types

#-------------------------------------------------#

def prepare_spark_data(data_type, part_num=200, max_num=2000):
    DATA_DIR = 'data'
    dir_path = os.path.join(DATA_DIR, data_type)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    file_path = os.path.join(DATA_DIR, f'{data_type}.json.gz')

    part_i = 0
    with gzip.open(file_path) as fin:
        for group in (x for x in islice(fin, max_num)):
            if not part_i % part_num:
                part_file = f'part-{part_i // part_num:05d}.json.gz'
                part_file_path = os.path.join(dir_path, part_file)

            part_i += 1
            with gzip.open(part_file_path, 'ab') as fout:
                fout.write(group)
                

#-------------------------------------------------#
# helper functions for parsing raw data

def _get(item, key):
    """
    get values from dictionary
    """
    if isinstance(item, list):
        return [i.get(key) for i in item]
    elif isinstance(item, dict):
        return item.get(key)
    else:
        return None


def _int(num):
	if num is not 0:
		return int(num)
	else:
		return None


def _float(num):
    if num is not 0:
        return float(num)
    else:
        return None


def _fromtimestamp(timestamp):
    try:
        return datetime.fromtimestamp(timestamp // 1000, 
        						      timezone('UTC'))
    except TypeError:
        return None


def clean_text(doc):
    """
    text_cleaning
    """
    if doc:
	    # Remove HTML tags
	    words_only = re.sub('<[^<]+?>|[^a-zA-Z]', ' ', doc)
	    # (only include words with more than 3 characters)
	    words = [word for word in words_only.lower().split() if len(word) > 3]
	    return " ".join(words)

#-------------------------------------------------#
# 

class Group():
    KEYS = ['id', 'city', 'state', 'timezone', 'lon', 'lat',
                  'rating', 'description', 'members', 'created',
                  'name', 'organizer', 'category', 'who', 'topics',
                  'link', 'group_photo']

    def __init__(self, group_id, city, state, timezone, lon, lat,
                 rating, description, members, created, name,
                 organizer, category, who, topics, link, group_photo):
        self.id = group_id
        self.city = city
        self.state = state
        self.timezone = timezone
        self.who = who
        self.link = link
        self.name = clean_text(name)
        self.lon = _float(lon)
        self.lat = _float(lat)
        self.rating = _float(rating)
        self.members = _int(members)
        self.created = _fromtimestamp(created)
        self.description = clean_text(description)
        self.organizer = _get(organizer, 'member_id')
        self.category = _get(category, 'id')
        self.tags = _get(topics, 'urlkey')
        self.group_photo_link = _get(group_photo, 'photo_link')

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)


class Event():
    KEYS = ['id', 'name', 'venue', 'rating', 'event_hosts',
            'yes_rsvp_count', 'maybe_rsvp_count', 'waitlist_count',
            'description', 'group', 'created', 'time', 'updated']

    def __init__(self, event_id, name, venue, rating, hosts,
                 yes_rsvp_count, maybe_rsvp_count, waitlist_count,
                 description, group, created, time, updated):
        self.id = event_id
        self.name = clean_text(name)
        self.group_id = _get(group, 'id')
        self.venue_id = _get(venue, 'id')
        self.rating_count = _int(_get(rating, 'count'))
        self.rating = _float(_get(rating, 'average'))
        self.yes_rsvp = _int(yes_rsvp_count)
        self.maybe_rsvp = _int(maybe_rsvp_count)
        self.waitlist = _int(waitlist_count)
        self.description = clean_text(description)
        self.hosts = _get(hosts, 'member_id')
        self.time = _fromtimestamp(time)
        self.created = _fromtimestamp(created)
        self.updated = _fromtimestamp(updated)
        self.created_tstamp = created // 1000
        self.time_tstamp = time // 1000

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)


class Venue():
    KEYS = ['id', 'name', 'rating', 'rating_count',
            'lon', 'lat', 'address_1', 'zip', 'city', 'state']

    def __init__(self, venue_id, name, rating, rating_count,
                 lon, lat, address_1, zip_code, city, state):
        self.id = venue_id
        self.name = clean_text(name)
        self.rating = _float(rating)
        self.rating_count = _int(rating_count)
        self.lon = _float(lon)
        self.lat = _float(lat)
        self.address = address_1
        self.zip = zip_code
        self.city = city
        self.state = state

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)


class Rsvp():
    KEYS = ['rsvp_id', 'response', 'member', 'event',
            'venue', 'group', 'created', 'mtime']

    def __init__(self, rsvp_id, response, member,
                 event, venue, group, created, mtime):
        self.id = rsvp_id
        self.response = response
        self.member_id = _get(member, 'member_id')
        self.event_id = _get(event, 'id')
        self.group_id = _get(group, 'id')
        self.venue_id = _get(venue, 'id')
        self.created = _fromtimestamp(created)
        self.mtime = _fromtimestamp(mtime)
        self.created_tstamp = created // 1000
        self.mtime_tstamp = mtime // 1000

    @classmethod
    def parse(cls, line):
        json_object = json.loads(line)
        vals = [json_object.get(k) for k in cls.KEYS]
        return cls(*vals)

#-------------------------------------------------#
# make Spark RDD

def localpath(path):
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
    return 'file://' + os.path.join(base_path, 'data', path)


def make_rdd(sc, data_type=None):
    """
    make spark rdd
    """
    if data_type == 'group':
        groupsRDD = sc.textFile(localpath('groups')) \
                      .flatMap(lambda line: line.split("\n")) \
                      .map(Group.parse) \
                      .filter(lambda grp: grp.category is not None) \

        return groupsRDD

    elif data_type == 'event':
        eventsRDD = sc.textFile(localpath('events')) \
                      .flatMap(lambda line: line.split("\n")) \
                      .map(Event.parse)

        return eventsRDD

    elif data_type == 'venue':
        venuesRDD = sc.textFile(localpath('venues')) \
                      .flatMap(lambda line: line.split("\n")) \
                      .map(Venue.parse)

        return venuesRDD

    elif data_type == 'rsvp':
        rsvpsRDD = sc.textFile(localpath('rsvps')) \
                     .flatMap(lambda line: line.split("\n")) \
                     .map(Rsvp.parse)
        return rsvpsRDD
    
#-------------------------------------------------#
# make Spark DataFrame

cols_dict = {
      'group': (("group_id", 'int'), ("name", 'str'), ("city", 'str'), ("state", 'str'),
              ("timezone", 'str'), ("created", 'dt'), ("members", 'int'),
              ('organizer', 'int'), ("tags", 'arr_str'), ("topic", 'str'), ("rating", 'float'),
              ("description", 'str'), ("link", 'str'), ("group_photo_link", 'str')), 
      'event': (("event_id", 'str'), ("time", "dt"), ("yes_rsvp", 'int'),
              ("rating", 'float'), ("rating_count", 'int'), ('hosts', 'arr_int'),
              ("group_id", 'int'), ("venue_id", 'int'), ('event_create_stamp', 'int'), ('event_time_stamp', 'int')),
      'venue': (('venue_id', 'int'), ('name', 'str'), ('city', 'str'), ('state', 'str'),
              ('lat', 'float'), ('lon', 'float'), ('zip', 'str'), ('rating', 'float'), ('rating_count', 'int')),
      'rsvp': (('rsvp_id', 'int'), ('created', 'dt'), ('mtime', 'dt'), ('response', 'str'),
              ('member_id', 'int'), ('event_id', 'str'), ('venue_id', 'int'),
              ('rsvp_create_stamp', 'int'), ('rsvp_mtime_stamp', 'int')) 
}

def get_schema(data_type=None):
    type_lib = {
        'int': types.IntegerType(),
        'float': types.FloatType(),
        'str': types.StringType(),
        'dt': types.TimestampType(),
        'arr_int': types.ArrayType(types.IntegerType()),
        'arr_float': types.ArrayType(types.FloatType()),
        'arr_str': types.ArrayType(types.StringType()),
    }

    return types.StructType([
      types.StructField(fname, type_lib[ftype])
      for fname, ftype in cols_dict[data_type]
  ])


def make_DF(sc, data_type=None,):
    sqlc = SQLContext(sc)
    rdd = make_rdd(sc, data_type=data_type)
    
    if data_type == 'group':
        topic_mapping = load_category_topic_mapping()
        rdd = rdd.map(lambda grp: (grp.id, grp.name, grp.city, grp.state,
                            grp.timezone, grp.created, grp.members,
                            grp.organizer, grp.tags, topic_mapping[grp.category], grp.rating,
                            grp.description, grp.link, grp.group_photo_link))
    elif data_type == 'event':
        rdd = rdd.map(lambda evt: (evt.id, evt.time, evt.yes_rsvp,
                            evt.rating, evt.rating_count, evt.hosts, evt.group_id,
                            evt.venue_id, evt.created_tstamp, evt.time_tstamp))

    elif data_type == 'venue':
        rdd = rdd.map(lambda vnu: (vnu.id, vnu.name, vnu.city, vnu.state,
                        vnu.lat, vnu.lon, vnu.zip, vnu.rating, vnu.rating_count))

    elif data_type == 'rsvp':
        rdd = rdd.map(lambda rsvp: (rsvp.id, rsvp.created, rsvp.mtime, rsvp.response,
                        rsvp.member_id, rsvp.event_id, rsvp.venue_id,
                        rsvp.created_tstamp, rsvp.mtime_tstamp))

    schema = get_schema(data_type=data_type)
    return sqlc.createDataFrame(rdd, schema).cache()