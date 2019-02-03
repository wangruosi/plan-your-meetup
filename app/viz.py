"""
Visualization
"""
from .helper import (load_cities,
                     load_topics,
                     load_city_coordinates)

import os
import numpy as np
import pandas as pd
from itertools import chain
from scipy import stats

from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, LinearColorMapper, Span
from bokeh.transform import jitter
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.palettes import Category10_3, Category10, Category20, Reds8


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'results'))
occurrance_path = os.path.join(base_path, 'descriptive', 'occurrance')
venues_path = os.path.join(base_path, 'descriptive', 'venues')

DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
TIME = {9: '9 am', 12: '12 pm', 15: '3 pm', 18: '6 pm', 21: '9 pm'}
weekday_mapping = dict(zip(list(range(0, 7)), DAYS))


# ------------------------------------------------- #

def _load_cities_pct(cities):
    """
    """
    file = os.path.join(occurrance_path, 'yearly')
    df = pd.read_pickle(file).sum(axis=1).unstack(-2).fillna(0)
    pct = pd.DataFrame((
        df.values * df.sum().sum() /
        df.sum(0).values.reshape(1, -1) /
        df.sum(1).values.reshape(-1, 1)
    ), columns=df.columns, index=df.index)
    return pct.loc[:, cities]


def plot_popular(city, data):
    """
    bar plots
    """
    source = ColumnDataSource(data=data)
    p = figure(x_range=data['topics'],
               y_range=[0, 4],
               plot_height=250,
               plot_width=200,
               toolbar_location=None,
               title=city)
    p.vbar(x='topics', top='pct', width=0.8, source=source)
    hline = Span(location=1, dimension='width', line_width=1)
    p.renderers.extend([hline])

    p.grid.grid_line_color = None
    p.axis.major_tick_line_color = None
    p.xaxis.major_label_orientation = np.pi / 4
    p.axis.major_label_standoff = 0
    p.axis.major_label_text_font_size = "10pt"
    p.yaxis.ticker = [1, 2, 3]
    p.yaxis.major_label_overrides = {1: 'Average', 2: '200%', 3: '300%'}

    return p


def city_popular_topics(cities):
    """
    plot the most distinctively popular topics
    """
    city_pct = _load_cities_pct(cities)
    plots = []
    for i, city in enumerate(cities):
        df = city_pct[city].sort_values(ascending=False)[:5]
        data = dict(topics=list(df.index),
                    pct=list(df.values))
        plots.append(plot_popular(city, data))
    return row(plots)


# ------------------------------------------------- #


def _load_city_topic(city, topic):
    data_path = os.path.join(occurrance_path)
    return pd.read_pickle(os.path.join(data_path, f'{city}_{topic}'))


def event_occurrance(city, topic, years=[2015, 2016, 2017]):
    df = _load_city_topic(city, topic)
    df = df.loc[df.year.isin(years), ['local_time', 'weekday']]
    data = dict(day=df['weekday'].map(weekday_mapping),
                time=df['local_time'].dt.hour +
                df['local_time'].dt.minute / 60)
    source = ColumnDataSource(data)

    p = figure(plot_width=600, plot_height=300,
               y_range=list(reversed(DAYS)), x_range=[6, 24],
               title=f"Events Occurance | {topic} meetups ({years[0]}—{years[-1]})")
    p.circle(x=jitter('time', width=1),
             y=jitter('day', width=0.6, range=p.y_range),
             source=source, alpha=0.1)

    p.ygrid.grid_line_color = None
    p.yaxis.major_tick_line_color = None
    p.xaxis.ticker = list(TIME.keys())
    p.xaxis.major_label_overrides = TIME
    p.xaxis.axis_label = 'Hour'
    p.yaxis.axis_label = 'Day'
    return p

# ------------------------------------------------- #


def _get_daily_occurance(city, topic_N=10):
    """
    load data
    """
    file = os.path.join(occurrance_path, 'daily')
    df = pd.read_pickle(file).loc[city]
    top_topics = df.sum(1).sort_values(ascending=False)[:topic_N].index
    occurance_df = df.loc[top_topics]
    for topic in top_topics:
        occurance_df.loc[topic] = df.loc[topic] / df.loc[topic].sum()
    return occurance_df


def daily_occurrance(city):
    """
    plot events daily occurance
    """
    df = _get_daily_occurance(city).stack(1).reset_index()
    topics = list(df.topic.unique())
    df['weekday'] = df.weekday.map(weekday_mapping)

    mapper = LinearColorMapper(palette=list(reversed(Reds8)),
                               low=df['count'].min(),
                               high=df['count'].max())

    p = figure(title='Daily Occurance',
               y_range=list(reversed(DAYS)),
               x_range=topics,
               plot_width=len(topics) * 40, plot_height=7 * 50,
               tools="", tooltips=[('percentage', '@count')])

    p.grid.grid_line_color = 'white'
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 4

    p.rect(x="topic", y="weekday", width=1, height=1,
           source=df,
           fill_color={'field': 'count', 'transform': mapper},
           line_color=None)
    return p

# ------------------------------------------------- #
#


def _get_hourly_occurance(city, topics, weekday_type=None):
    assert city in load_cities()
    assert set(topics).issubset(set(load_topics()))

    hour_range = 24
    file = os.path.join(occurrance_path, f'hourly_{weekday_type}')

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


def hourly_occurrance(city, topics):
    topic_n = len(topics)
    colors = _get_colors(topic_n)

    # data
    sources = []
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
                plot_width=400, plot_height=380,
                title="Hourly Occurrance | Workdays", tools="")
    p2 = figure(x_range=p1.x_range, y_range=p1.y_range,
                plot_width=400, plot_height=380,
                title="Hourly Occurrance | Weekends", tools="")

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
            p.xaxis.ticker = list(TIME.keys())
            p.xaxis.major_label_overrides = TIME

    return row(p1, p2)

# ------------------------------------------------- #
# Explore Venues


def _wgs84_to_web_mercator(df, lon="lon", lat="lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat].astype(float)) * np.pi / 360.0)) * k
    return df


def explore_venues(city, topics):
    colors = _get_colors(len(topics))

    df_city = pd.DataFrame(load_city_coordinates()).T
    df_city.columns = ['lat', 'lon']

    file = os.path.join(venues_path, f'{city}_venues')
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
    for topic, color in zip(topics, colors):
        source = ColumnDataSource(df_venues.loc[df_venues.topic == topic])
        p.circle('x', 'y',
                 line_alpha=0.05,
                 fill_color=color,
                 legend=topic,
                 size='host_num',
                 fill_alpha=0.2,
                 source=source)
    return p
