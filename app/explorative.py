"""
Explorative data analysis of events occurrence
and venue locations.

Ruosi Wang ruosiwang.psy@gmail.com
"""
from helper import load_cities, get_path
from spark_data import make_DF

import os
import numpy as np
from datetime import datetime
from pytz import timezone

from pyspark import SparkContext
from pyspark.sql import types
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf


occurrence_path = get_path('results', 'explorative', 'occurrence')
venues_path = get_path('results', 'explorative', 'venues')
# ------------------------------------------------- #
# helper functions for transforming datetime


def floor_req(val):
    if val >= 2:
        return 1
    else:
        return 0


def timezone_convert(dt, tz):
    return dt.astimezone(timezone(tz))


def get_median(list_nums):
    try:
        median = np.nanmedian(list_nums)
        return int(round(median))
    except Exception:
        return None


# udf functions
udf_floor_req = udf(floor_req, types.IntegerType())
udf_tzconvert = udf(timezone_convert, types.TimestampType())
udf_year = udf(lambda dt: dt.year, types.IntegerType())
udf_month = udf(lambda dt: dt.month, types.IntegerType())
udf_weekday = udf(lambda dt: dt.weekday(), types.IntegerType())
udf_hour = udf(lambda dt: dt.hour, types.IntegerType())
udf_active = udf(lambda dt: dt > datetime(2018, 1, 1))
udf_median = F.udf(get_median, types.IntegerType())

# ------------------------------------------------- #

if __name__ == '__main__':
    sc = SparkContext("local[*]", "temp")

    df_group = make_DF(sc, 'group')
    df_event = make_DF(sc, 'event')
    df_venue = make_DF(sc, 'venue')

    # ------------------------------------------------- #
    # Events occurrence

    event_time_info = df_event \
        .select(col('event_id'), col('group_id'), col('time')) \
        .join(df_group.select(col('group_id'), col('city'),
                              col('topic'), col('timezone')), 'group_id').cache()

    # convert to local time
    event_time_info = event_time_info \
        .withColumn('local_time', udf_tzconvert('time', 'timezone')) \
        .withColumn('year', udf_year('local_time')) \
        .withColumn('weekday', udf_weekday('local_time')) \
        .withColumn('hour', udf_hour('local_time')).cache()

    # year trends
    year_group = ['city', 'topic', 'year']
    yearly_occurrence = event_time_info \
        .groupBy(year_group) \
        .count().sort(year_group) \
        .toPandas().set_index(year_group) \
        .unstack(-1).fillna(0).astype(int)
    yearly_occurrence.to_pickle(os.path.join(occurrence_path, 'yearly'))

    # daily occurrence
    weekday_group = ['city', 'topic', 'weekday']
    daily_occurrence = event_time_info \
        .groupBy(weekday_group) \
        .count().sort(weekday_group) \
        .toPandas().set_index(weekday_group) \
        .unstack(-1).fillna(0).astype(int)
    daily_occurrence.to_pickle(os.path.join(occurrence_path, 'daily'))

    # hourly occurrence
    hour_group = ['city', 'topic', 'hour']
    weekday_hourly_occurrence = event_time_info \
        .filter(event_time_info.weekday < 6) \
        .groupBy(hour_group).count().sort(hour_group) \
        .toPandas().set_index(hour_group) \
        .unstack(-1).fillna(0).astype(int)
    weekday_hourly_occurrence.to_pickle(os.path.join(occurrence_path, 'hourly_weekday'))

    weekend_hourly_occurrence = event_time_info \
        .filter(event_time_info.weekday > 5) \
        .groupBy(hour_group).count().sort(hour_group) \
        .toPandas().set_index(hour_group) \
        .unstack(-1).fillna(0).astype(int)
    weekend_hourly_occurrence.to_pickle(os.path.join(occurrence_path, 'hourly_weekend'))

    # ------------------------------------------------- #
    # Venues
    venue_info = df_venue \
        .select(col('venue_id'), col('lat'), col('lon'),
                col('name').alias('venue'), col('rating')).cache()

    distinct_venue = df_venue \
        .groupBy(['lat', 'lon']) \
        .agg(F.min('venue_id').alias('loc_id')).cache()

    group_info = df_group \
        .select(col('group_id'), col('topic'),
                col('name').alias('group'), col('city')).cache()

    event_venue = df_event \
        .select(['group_id', 'venue_id']) \
        .join(venue_info, 'venue_id').cache()

    group_venue = event_venue \
        .groupby(['group_id', 'lat', 'lon']) \
        .agg(F.count('*').alias('host_num')) \
        .join(distinct_venue, ['lat', 'lon']).cache()

    group_venue_info = group_venue \
        .join(group_info, 'group_id') \
        .withColumnRenamed('loc_id', 'venue_id') \
        .join(venue_info.select(['venue_id', 'venue', 'rating']), 'venue_id').cache()

    group_venue_info_floor = group_venue_info \
        .withColumn('host', udf_floor_req('host_num')).cache()

    # group <-> venue
    cities = load_cities()

    for city in cities:
        df = group_venue_info \
            .filter((col("host_num") > 3) & (col("city") == city)) \
            .select(['lat', 'lon', 'topic', 'host_num']).toPandas()
        df.to_pickle(os.path.join(venues_path, f'{city}_venues'))

    # ------------------------------------------------- #
    # to-do: RSVP patterns
