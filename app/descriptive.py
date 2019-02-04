"""
Explorative data analysis of events occurance
and venue locations.

Ruosi Wang ruosiwang.psy@gmail.com
"""

from spark_data import make_DF

import os
import pandas as pd
from datetime import datetime
from pytz import timezone

from pyspark import SparkContext
from pyspark.sql import SQLContext, types
import pyspark.sql.functions as F
from pyspark.sql.functions import col, udf

#-------------------------------------------------#
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

#-------------------------------------------------#

if __name__ == '__main__':
    sc = SparkContext("local[*]", "temp")

    df_group = make_DF(sc, 'group')
    df_event = make_DF(sc, 'event')
    df_venue = make_DF(sc, 'venue')
    
    #-------------------------------------------------#
    # Popular Topics
    
    
    #-------------------------------------------------#
    # Events occurance
    
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
    year_trends = event_time_info \
        .groupBy(year_group) \
        .count().sort(year_group) \
        .toPandas().set_index(year_group) \
        .unstack(-1).fillna(0).astype(int)
    
    # daily occurance
    weekday_group = ['city', 'topic', 'weekday']
    daily_occurance = event_time_info \
        .groupBy(weekday_group) \
        .count().sort(weekday_group) \
        .toPandas().set_index(weekday_group) \
        .unstack(-1).fillna(0).astype(int)

    # hourly occurance
    hour_group = ['city', 'topic', 'hour']
    workday_hourly_occurance = event_time_info \
        .filter(event_time_info.weekday < 6) \
        .groupBy(hour_group).count().sort(hour_group) \
        .toPandas().set_index(hour_group) \
        .unstack(-1).fillna(0).astype(int)
  
    weekend_hourly_occurance = event_time_info \
        .filter(event_time_info.weekday > 5) \
        .groupBy(hour_group).count().sort(hour_group) \
        .toPandas().set_index(hour_group) \
        .unstack(-1).fillna(0).astype(int)

    # save pandas dataframe
    # daily_occurance.to_pickle(localpath("df_daily_occurance"))
    # workday_hourly_occurance.to_pickle(localpath("df_workday_hourly_occurance"))
    # weekend_hourly_occurance.to_pickle(localpath("df_weekend_hourly_occurance"))
    
    #-------------------------------------------------#
    # RSVP patterns
    
    #-------------------------------------------------#
    # Venues
