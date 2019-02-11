"""
Visualizations of'explorative analysis
Ruosi Wang ruosiwang.psy@gmail.com
"""
from helper import (load_cities,
                    load_topics,
                    load_city_coordinates)

import os
import numpy as np
import pandas as pd
from itertools import chain
from scipy import stats

from bokeh.layouts import row, column
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import (ColumnDataSource, LinearColorMapper, Span,
                          BasicTicker, PrintfTickFormatter, ColorBar)
from bokeh.transform import jitter
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.palettes import Category10_3, Category10, Category20, Reds8


base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", 'results'))
occurrence_path = os.path.join(base_path, 'explorative', 'occurrence')
venues_path = os.path.join(base_path, 'explorative', 'venues')

DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
TIME = {9: '9 am', 12: '12 pm', 15: '3 pm', 18: '6 pm', 21: '9 pm'}
weekday_mapping = dict(zip(list(range(0, 7)), DAYS))

topics = load_topics()
colors = Category20[20] + ['gray'] * (len(topics) - 10)
color_mapping = dict(zip(topics, colors))

output_notebook()
# ------------------------------------------------- #


def _load_us_pct():
    """
    """
    file = os.path.join(occurrence_path, 'yearly')
    df = pd.read_pickle(file).sum(axis=1).unstack(-2).fillna(0)
    return df.sum(axis=1) * 100 / df.sum().sum()


def _load_cities_pct(cities, normalize=False):
    """
    """
    file = os.path.join(occurrence_path, 'yearly')
    df = pd.read_pickle(file).sum(axis=1).unstack(-2).fillna(0)

    if normalize:
        data = (
            df.values * df.sum().sum() * 100 /
            df.sum(0).values.reshape(1, -1) /
            df.sum(1).values.reshape(-1, 1)
        )
    else:
        data = df.values * 100 / df.values.sum(axis=0)

    pct = pd.DataFrame(data, columns=df.columns, index=df.index)
    return pct.loc[:, cities]


def plot_popular(city, data, normalize=False, y_range=[]):
    """
    bar plots
    """
    topic_n = len(data['topics'])
    data['color'] = list(map(lambda x: color_mapping[x], data['topics']))
    source = ColumnDataSource(data)

    p = figure(x_range=data['topics'],
               y_range=y_range,
               plot_height=300,
               plot_width=50 * topic_n,
               toolbar_location=None,
               title=city, tooltips='$y{1.1}%')
    p.vbar(x='topics', top='pct', color='color', width=0.8, source=source)

    p.grid.grid_line_color = None
    # p.axis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.major_label_orientation = np.pi / 4
    p.axis.major_label_standoff = 0
    p.axis.major_label_text_font_size = "10pt"

    if normalize:
        p.yaxis.ticker = [100, 200, 300]
        p.yaxis.major_label_overrides = {100: 'Average', 200: '200%', 300: '300%'}
        hline = Span(location=1, dimension='width', line_width=1)
        p.renderers.extend([hline])
    else:
        p.yaxis.ticker = np.linspace(*y_range, num=5)

    return p


def plot_cities_popular_topics(cities, topic_n=5, normalize=False):
    """
    plot the most popular topics in a list of cities
    """
    city_pct = _load_cities_pct(cities, normalize=normalize)
    if normalize:
        y_range = [0, 400]
    else:
        y_range = [0, np.round(city_pct.max().max() * 1.2)]

    plots = []
    for i, city in enumerate(cities):
        df = city_pct[city].sort_values(ascending=False)[:topic_n]
        data = dict(topics=list(df.index),
                    pct=list(df.values))
        plots.append(plot_popular(city, data,
                                  normalize=normalize,
                                  y_range=y_range))
    plots[0].yaxis.axis_label = 'percentage (%)'
    plots[0].plot_width += 20

    show(row(plots))


def plot_us_popular_topics(topic_n=10):
    """
    plot the most popular topics in the US
    """
    city_pct = _load_us_pct()
    y_range = [0, np.round(city_pct.max().max() * 1.2)]
    df = city_pct.sort_values(ascending=False)[:topic_n]
    data = dict(topics=list(df.index),
                pct=list(df.values))
    p = plot_popular('US 30 cities', data, y_range=y_range)
    p.yaxis.axis_label = 'percentage (%)'
    show(p)

# ------------------------------------------------- #


def _load_city_topic(city, topic):
    data_path = os.path.join(occurrence_path, 'city_topic')
    return pd.read_pickle(os.path.join(data_path, f'{city}_{topic}'))


def plot_event_occurrence(city, topic, years=[]):
    df = _load_city_topic(city, topic)
    df = df.loc[df.year.isin(years), ['local_time', 'weekday']]
    data = dict(day=df['weekday'].map(weekday_mapping),
                time=df['local_time'].dt.hour +
                df['local_time'].dt.minute / 60)
    source = ColumnDataSource(data)

    p = figure(plot_width=600, plot_height=300,
               y_range=list(reversed(DAYS)), x_range=[6, 24],
               title=f'{city} | {topic}')
    p.circle(x=jitter('time', width=1),
             y=jitter('day', width=0.6, range=p.y_range),
             source=source, alpha=0.1)

    hline = Span(location=2, dimension='width', line_width=0.5,
                 line_color='gray')
    p.renderers.extend([hline])

    p.ygrid.grid_line_color = None
    p.yaxis.major_tick_line_color = None
    p.xaxis.ticker = list(TIME.keys())
    p.xaxis.major_label_overrides = TIME
    p.xaxis.axis_label = 'Hour'
    p.yaxis.axis_label = 'Day'
    return p


def plot_compare_occurrence(city, topics, years=[2015, 2016, 2017]):
    p1 = plot_event_occurrence(city, topics[0], years=years)
    p2 = plot_event_occurrence(city, topics[1], years=years)
    p1.xaxis.axis_label = None
    show(column(p1, p2))

# ------------------------------------------------- #


def _get_daily_occurrence(city, topic_N=10):
    """
    load data
    """
    file = os.path.join(occurrence_path, 'daily')
    df = pd.read_pickle(file)

    if city == 'US':
        df = df.groupby('topic').sum()
    elif city in load_cities():
        df = pd.loc[city]

    top_topics = df.sum(1).sort_values(ascending=False)[:topic_N].index
    occurrence_df = df.loc[top_topics]
    for topic in top_topics:
        occurrence_df.loc[topic] = df.loc[topic] * 100 / df.loc[topic].sum()

    # sorted by event occurrence rates on weekends
    occurrence_df['weekend_prop'] = (occurrence_df.values[:, 5:].sum(axis=1) /
                                     occurrence_df.sum(axis=1))
    occurrence_df.sort_values('weekend_prop', inplace=True)
    occurrence_df.drop(columns=['weekend_prop'], inplace=True)

    return occurrence_df


def plot_daily_occurrence(city, topic_N=10):
    """
    plot events daily occurrence
    """
    df = _get_daily_occurrence(city, topic_N=topic_N)
    df = df.stack(1).reset_index()
    topics = list(df.topic.unique())
    df['weekday'] = df.weekday.map(weekday_mapping)

    colors = list(reversed(Reds8))
    mapper = LinearColorMapper(palette=colors,
                               low=df['count'].min(),
                               high=df['count'].max())

    p = figure(title='Daily Occurance',
               y_range=list(reversed(DAYS)),
               x_range=topics,
               plot_width=len(topics) * 40 + 100, plot_height=7 * 50,
               tools="", tooltips=[('percentage', '@count{1.1}%')])

    hline = Span(location=2, dimension='width', line_width=3, line_color='white')
    p.renderers.extend([hline])

    p.grid.grid_line_color = 'white'
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "10pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = np.pi / 3

    p.rect(x="topic", y="weekday", width=1, height=1,
           source=df,
           fill_color={'field': 'count', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="8pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%.0f%%"),
                         label_standoff=8, border_line_color=None, location=(2, 0))
    p.add_layout(color_bar, 'right')

    show(p)

# ------------------------------------------------- #
#


def _get_hourly_occurrence(city, topics, weekday_type=None):

    hour_range = 24
    file = os.path.join(occurrence_path, f'hourly_{weekday_type}')

    df = pd.read_pickle(file)
    if city == 'US':
        df = df.groupby('topic').sum()
    elif city in load_cities():
        df = pd.loc[city]
    occurrence_dict = {topic: list(chain(*[[h] * df.loc[topic][h]
                                           for h in range(hour_range)]))
                       for topic in topics}
    return occurrence_dict


def _gauss_kde(X, data, kde_factor=0.6):
    kde = stats.gaussian_kde(data)
    kde.set_bandwidth(bw_method='silverman')
    kde.set_bandwidth(bw_method=kde.factor / kde_factor)
    y = kde(X).tolist()
    return y


def _get_colors(topic_n):
    if topic_n < 3:
        return Category10_3[:topic_n]
    elif 2 < topic_n <= 10:
        return Category10.get(topic_n)
    else:
        return Category20.get(topic_n)


def plot_hourly_occurrence(city, topics):
    topic_n = len(topics)
    colors = _get_colors(topic_n)

    # data
    sources = []
    for i, weekday_type in enumerate(('weekday', 'weekend')):
        data = _get_hourly_occurrence(city, topics,
                                      weekday_type=weekday_type)
        hours = np.linspace(6, 24, 90)

        source = {key: _gauss_kde(hours, val) for key, val in data.items()}
        source['x'] = hours
        sources.append(source)

    y_max = max([max(source.get(topic)) for topic in topics
                 for source in sources]) * 1.2

    p1 = figure(x_range=[6, 24], y_range=[0, y_max],
                plot_width=400, plot_height=380,
                title="Hourly Occurrance | Weekdays", tools="")
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

    p2.yaxis.axis_label = None
    show(row(p1, p2))

# ------------------------------------------------- #
# Explore Venues


def _wgs84_to_web_mercator(df, lon="lon", lat="lat"):
    """Converts decimal longitude/latitude to Web Mercator format"""
    k = 6378137
    df["x"] = df[lon] * (k * np.pi / 180.0)
    df["y"] = np.log(np.tan((90 + df[lat].astype(float)) * np.pi / 360.0)) * k
    return df


def plot_venues(city, topics=[], scope=10000, alpha=0.2,
                x_move=0, y_move=0, legend_location='bottom_right'):
    colors = ['#ea4335', '#fbbc05', '#34a853', '#673ab7']

    df_city = pd.DataFrame(load_city_coordinates()).T
    df_city.columns = ['lat', 'lon']

    file = os.path.join(venues_path, f'{city}_venues')
    df_venues = pd.read_pickle(file)
    df_venues = _wgs84_to_web_mercator(df_venues)
    df_venues['host_num'] = df_venues['host_num'].apply(lambda x: np.log(x) * 3)

    city_x, city_y = _wgs84_to_web_mercator(df_city.loc[city])['x':'y']

    X_range = (city_x - scope + x_move, city_x + scope + x_move)
    Y_range = (city_y - scope + y_move, city_y + scope + y_move)

    p = figure(x_range=X_range, y_range=Y_range, title=city,
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
                 fill_alpha=alpha,
                 source=source)

    p.title.text_font_size = '12pt'
    p.legend.location = legend_location

    show(p)
