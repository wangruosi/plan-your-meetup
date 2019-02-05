"""
Visualizations of topic classifications
Ruosi Wang ruosiwang.psy@gmail.com
"""

from helper import get_path, load_topics
from sklearn_models import most_important_features

import pickle
import math
import pandas as pd
import numpy as np
from itertools import product
from collections import OrderedDict
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

import matplotlib.pyplot as plt
from bokeh.palettes import Reds8
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, LinearColorMapper, Span,
                          BasicTicker, PrintfTickFormatter, ColorBar)
from bokeh.core.properties import value
from bokeh.transform import dodge



# ------------------------------------------------- #
# word cloud of meetup topics

def _get_lemma(words):
    lemma_dict = dict()
    lemmatizer = WordNetLemmatizer()
    for i, word in enumerate(words):
        lemma = lemmatizer.lemmatize(word)
        if lemma not in lemma_dict:
            lemma_dict[lemma] = len(words) + i
    return lemma_dict

def _cal(topic_num):
    return (topic_num // 5 + (topic_num % 5 != 0) ) * 5

def _get_top_lemma(topic_num, feature_num):
    file = get_path('results', 'models', f'MNB_{topic_num:02d}')
    model = pickle.load(open(file, 'rb'))
    topics = load_topics()[:topic_num]
    
    top_words = most_important_features(model, feature_num)
    lemma_dict = OrderedDict()
    for topic, words in zip(topics, top_words):
        lemma_dict[topic] = _get_lemma(words)
    return lemma_dict
    

def plot_word_cloud(topic_num=5, feature_num=25, row_num=None):
    """
    plot critical features as word cloud
    """
    lemma_dict = _get_top_lemma(_cal(topic_num), feature_num)
    
    col_num, plot_size = 3, 4
    row_num = math.ceil(topic_num / col_num)
    
    fig, axs = plt.subplots(row_num, col_num, 
                            figsize=(col_num * plot_size, 
                                     row_num * plot_size))
    axs = axs.ravel()
    
    sbp_info = list(zip(np.tile(np.array(range(row_num)), (col_num, 1)).T.flatten(),
                        np.tile(np.array(range(col_num)), row_num)))

    for idx, topic in enumerate(lemma_dict):
        if idx >= topic_num:
            break
        
        word_cloud = WordCloud(width=400, height=400,
                               max_font_size=80,
                              background_color="white")
        word_cloud.generate_from_frequencies(lemma_dict[topic])
        # i, j = sbp_info[idx]
        axs[idx].imshow(word_cloud, interpolation='bilinear')
        axs[idx].set_title(topic, size=20)
    
    [axi.set_axis_off() for axi in axs]
    plt.show()
    
# ------------------------------------------------- #
# benchmark classification results 

def save_classifications(model_list, topic_nums):
    """
    get model classification results and save them as a txt file
    """
    
    tuples = list(product(topic_nums, model_list)) 
    index = pd.MultiIndex.from_tuples(tuples, names=['topic_num', 'model'])

    f1_results, acc_results = [], []
    for topic_num, model in tuples:
        file = get_path('results', 'classifications', 
                        f'{model}_{topic_num:02d}')
        f1, acc, _ = pickle.load(open(file, 'rb'))
        f1_results.append(f1)
        acc_results.append(acc)
    df = pd.DataFrame({'f1': f1_results, 'acc': acc_results}, index=index)
    file = get_path('results', 'classifications', 'summary.txt')
    
    df.to_csv(file)

def _load_classification():
    file = get_path('results', 'classifications', 'summary.txt')
    df = pd.read_csv(file, index_col=[0,1])
    return df

def plot_topic_classifications(metric='f1'):
    """
    bechmark topic classification results
    """
    scores = _load_classification()[metric].unstack()

    # # prepare bokeh data
    data = {m: list(scores[m]) for m in list(scores.columns)}
    data['topic'] = [f'{n}' for n in list(scores.index)]
    source = ColumnDataSource(data=data)
    colors = ['#ED6A5A', '#F4F1BB', '#9BC1BC', '#5CA4A9',
              '#E6EBE0', '#dcf5ff', '#e6c86e', '#508cd7']

    models = scores.T.sort_values([20], ascending=False).index[:5]
    dis = [-0.3, -0.15, 0, 0.15, 0.3]

    p = figure(x_range=data['topic'], y_range=(.6, 1), 
               plot_height=350, plot_width=550,
               title="Meetup Topic Classification",
               toolbar_location=None, tools="",
               tooltips="score: $y")

    for i in range(len(models)):
        p.vbar(x=dodge('topic', dis[i], range=p.x_range), 
               top=models[i], width=0.12, source=source,
               color=colors[i], legend=value(models[i]))

    p.x_range.range_padding = 0.1
    p.xgrid.grid_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.xaxis.axis_label = "Number of Topics to Classify"
    p.yaxis.axis_label = f'{metric} score'
    p.legend.location = "top_left"
    p.legend.orientation = "horizontal"

    return p

# ------------------------------------------------- #
# classification confusion matrix

def _load_conf_matrix(model_name):
    file = get_path('results', 'classifications', model_name)
    _, _,conf_mat = pickle.load(open(file, 'rb'))
    return conf_mat


def _conf_mat2df(conf_mat, topics):
    """
    convert confusion matrix to dataframe
    """
    normalized = conf_mat * 100 / conf_mat.sum(axis=0) 

    raw = pd.DataFrame(conf_mat, index = topics, columns = topics).stack()
    normalized = pd.DataFrame(normalized, index = topics, columns = topics).stack()

    df =  pd.concat([raw, normalized], axis=1).reset_index()
    df.columns = ['x', 'y', 'count', 'percentage']
    return df


def plot_conf_matrix(model_name, topic_num):
    """
    plot confusion matrix 

    """
    topics = load_topics()[:topic_num]
    file = f'{model_name}_{topic_num:02d}'
    conf_mat = _load_conf_matrix(file)
    df = _conf_mat2df(conf_mat, topics)
    
    
    colors = list(reversed(Reds8))
    mapper = LinearColorMapper(palette=colors,
                               low=0, high=20)

    p = figure(title=f'{model_name} | Confusion Matrix (Normalized)',
               y_range=list(reversed(topics)), x_range= topics,
               plot_width=500, plot_height= 400,
               tools="", tooltips=[('count', '@count' ),('percentage', '@percentage%')])


    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_standoff = 0
    p.axis.major_label_text_font_size = "10pt"
    p.xaxis.major_label_orientation = np.pi / 4

    p.rect(x="x", y="y", width=0.9, height=0.9,
           source=df,
           fill_color={'field': 'percentage', 'transform': mapper},
           line_color=None)

    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="5pt",
                         ticker=BasicTicker(desired_num_ticks=len(colors)),
                         formatter=PrintfTickFormatter(format="%f%%"),
                         label_standoff=8, border_line_color=None, location=(2, 0))
    p.add_layout(color_bar, 'right')
    p.yaxis.axis_label = 'Actural Topics'
    p.xaxis.axis_label = 'Predicted Topics'

    return p