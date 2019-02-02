"""
Visualization
"""
from .helper import (load_cities, 
                     load_topics,
                     load_city_coordinates)

import os
import pickle
import numpy as np
import pandas as pd
from itertools import chain
from scipy import stats

import matplotlib.pyplot as plt
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.transform import dodge
from bokeh.core.properties import value
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.palettes import Category10_3, Category10, Category20, Reds8


#-------------------------------------------------#
#
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
descriptive_path = os.path.join(base_path, 'results', 'descriptive')


def _get_daily_occurance(city, topics):
    assert city in load_cities()
    assert set(topics).issubset(set(load_topics()))

    file = os.path.join(descriptive_path, 'occurrance', 'daily')
    df = pd.read_pickle(file).loc[city].loc[topics]
    for topic in topics:
        df.loc[topic] = df.loc[topic] / df.loc[topic].sum()
    return df


def daily_occurance(city, topics):
    all_topics = load_topics()
    df = _get_daily_occurance(city, all_topics).stack(1).reset_index()

    mapper = LinearColorMapper(palette=list(reversed(Reds8)),
                               low=df['count'].min(),
                               high=df['count'].max())
    display_topics = topics + [topic for topic in all_topics if topic not in topics]
    p = figure(title='Weekly Occurance',
               y_range=list(reversed(display_topics)),
               x_range=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
               plot_width=7 * 50, plot_height=len(all_topics) * 20,
               tools="", tooltips=[('percentage', '@count')])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(y="topic", x="weekday", width=1, height=1,
           source=df,
           fill_color={'field': 'count', 'transform': mapper},
           line_color=None)
    return p


#-------------------------------------------------#
#

def _get_hourly_occurance(city, topics, weekday_type=None):
    assert city in load_cities()
    assert set(topics).issubset(set(load_topics()))

    hour_range = 24
    file = os.path.join(descriptive_path, 'occurrance', f'hourly_{weekday_type}')
    
    df = pd.read_pickle(file).loc[city]
    occurance_dict = {topic: list(chain(*[[h] * df.loc[topic][h]
                                          for h in range(hour_range)]))
                      for topic in topics}
    return occurance_dict


def _gauss_kde(X, data, kde_factor=0.6):
    kde = stats.gaussian_kde(data)
    kde.set_bandwidth(bw_method='silverman')
    kde.set_bandwidth(bw_method=kde.factor / kde_factor)
    y = kde(X).tolist()
    # y[0], y[-1] = 0, 0
    return y


def _get_colors(topic_n):
    if topic_n < 3:
        return Category10_3[:topic_n]
    elif 2 < topic_n <= 10:
        return Category10.get(topic_n)
    else:
        return Category20.get(topic_n)


def hourly_occurance(city, topics):
    topic_n = len(topics)
    colors = _get_colors(topic_n)

    # data
    ps, sources = [], []
    for i, weekday_type in enumerate(('weekday', 'weekend')):
        data = _get_hourly_occurance(city, topics,
                                    weekday_type=weekday_type)
        hours = np.linspace(6, 24, 90)

        source = {key: _gauss_kde(hours, val) for key, val in data.items()}
        source['x'] = hours
        sources.append(source)

    y_max = max([max(source.get(topic)) for topic in topics
                 for source in sources]) * 1.2

    p1 = figure(x_range=[6, 24], y_range=[0, y_max],
                plot_width=400, plot_height=400,
                title="Daily Occurance | Workdays", tools="")
    p2 = figure(x_range=p1.x_range, y_range=p1.y_range,
                plot_width=400, plot_height=400,
                title="Daily Occurance | Weekends", tools="")

    for i, p in enumerate((p1, p2)):
        for topic, color in zip(topics, colors):
            p.line(x=sources[i]['x'], y=sources[i][topic], color=color,
                   muted_color=color, muted_alpha=0.2,
                   line_width=3, legend=topic)

            p.legend.location = "top_left"
            p.legend.click_policy = "mute"

            p.grid.grid_line_alpha = 0.3
            p.xaxis.axis_label = 'Hour'
            p.yaxis.axis_label = 'Frequency'
            p.xaxis.ticker = [9, 12, 15, 18, 21]
            p.xaxis.major_label_overrides = {9: '9 am', 12: '12 pm',
                                             15: '3 pm', 18: '6 pm', 21: '9 pm'}

    return row(p1, p2)

#-------------------------------------------------#
#

def event_occurance(city, topics):
    weekly_p = weekly_occurance(city, topics)
    daily_p = daily_occurance(city, topics)
    return row(weekly_p, daily_p)

# --------------- explore venues ------------------------------------ #


def _wgs84_to_web_mercator(df, lon="lon", lat="lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat].astype(float)) * np.pi / 360.0)) * k
    return df


def explore_venues(city):
    color_palette = ['#ea4335', '#fbbc05', '#34a853', '#673ab7']
    categories = ['social', 'career-business', 'tech', 'music']

    df_city = pd.DataFrame(load_city_coordinates()).T
    df_city.columns = ['lat', 'lon']

    file = os.path.join(descriptive_path, 'venues', f'{city}_venues')
    df_venues = pd.read_pickle(file)
    df_venues = _wgs84_to_web_mercator(df_venues)
    df_venues['host_num'] = df_venues['host_num'].apply(lambda x: np.log(x) * 3)

    city_x, city_y = _wgs84_to_web_mercator(df_city.loc[city])['x':'y']

    X_range = (city_x - 10000, city_x + 10000)
    Y_range = (city_y - 10000, city_y + 10000)

    p = figure(x_range=X_range, y_range=Y_range,
               x_axis_type="mercator", y_axis_type="mercator")

    # load map
    p.add_tile(CARTODBPOSITRON)
    p.axis.visible = False
    for ctg, cp in zip(categories, color_palette):
        source = ColumnDataSource(df_venues.loc[df_venues.topic == ctg])
        p.circle('x', 'y',
                 line_alpha=0.05,
                 fill_color=cp,
                 legend=ctg,
                 size='host_num',
                 fill_alpha=0.3,
                 source=source)
    return p
