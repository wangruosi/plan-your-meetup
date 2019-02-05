"""
meetup documents topic classifier
@Ruosi Wang ruosiwang.psy@gmail.com

Example:

> python topic_classifier.py --model GRU --topic_num 10 

"""

from helper import get_path
from sklearn_models import build_SKM
from keras_models import (build_DNN,
                          set_callback,
                          get_embedding_matrix,
                          tokenizer_transform)

import json
import pickle
import argparse
import numpy as np

from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV

# load model details


def load_model_info():
    file = get_path('data', 'models_info.json')
    return json.load(open(file, 'r'))


# load data and prepare
def load_dataset(data_type, topic_num):
    file = get_path('data', 'documents', f'{data_type}_{topic_num:02d}')
    return pickle.load(open(file, 'rb'))


# model evaluation
def model_evaluation(X_test, y_test, model=None, doc2seq=None):

    is_DNN = model_name[:3] in ['MLP', 'GRU', 'CNN']
    # predictions
    if is_DNN:
        X_test = doc2seq(X_test)

    y_pred = model.predict(X_test)
    # data transform for dnn results
    if is_DNN:
        y_pred = np.argmax(y_pred, axis=1)

    # metrics
    f1 = f1_score(y_pred, y_test, average='weighted')
    acc = accuracy_score(y_pred, y_test)
    conf = confusion_matrix(y_pred, y_test)
    print(f'f1 score: {f1 * 100:.2f}%; accuracy: {acc * 100: .2f}%')
    # save
    return f1, acc, conf


def save_file(to_save, dir_name, file_name):
    file = get_path('results', dir_name, file_name)
    pickle.dump(to_save, open(file, 'wb'))


def load_file(dir_name, file_name):
    file = get_path('results', dir_name, file_name)
    pickle.load(open(file, 'rb'))


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MNB',
                        help='What is the model?')
    parser.add_argument('--topic_num', type=int, default=5,
                        help='What is the number of topics to classify?')
    parser.add_argument('--embedding_dim', type=int, default=50,
                        help='What is the number of embedding dim?')
    parser.add_argument('--max_seq_len', type=int, default=500,
                        help='What is the max sequence length for padding')
    parser.add_argument('--max_num_words', type=int, default=20000,
                        help='What is max number of words?')
    parser.add_argument('--epoch', type=int, default=10,
                        help='what is the number of epoch?')
    args = parser.parse_args()

    # load training documents data and model info
    X_train, y_train = load_dataset('train', topic_num=args.topic_num)
    X_test, y_test = load_dataset('test', topic_num=args.topic_num)
    models_info = load_model_info()

    # model details
    model_name = f'{args.model}_{args.topic_num:02d}'
    model_group = models_info[args.model]['group']
    params = models_info[args.model]['params']

    # model training
    print(f'{model_name} training...')

    if model_group == 'sklearn':
        # build models
        model = build_SKM(model_type=args.model,
                          max_features=20000,
                          selectK=10000)

        # grid search
        clf = GridSearchCV(model, params, cv=3, n_jobs=-1)
        clf.fit(X_train, y_train)

        # train with best params
        model.set_params(**clf.best_params_)
        model.fit(X_train, y_train)

        # test and save
        save_file(model, 'models', model_name)

        results = model_evaluation(X_test, y_test,
                                   model=model)

    if model_group == 'keras':
        # preprocessing
        X_train, word_index, doc2seq = tokenizer_transform(
            X_train, max_num_words=args.max_num_words,
            max_sequence_length=args.max_seq_len
        )
        y_train = to_categorical(y_train)

        # build models
        embedding_matrix = get_embedding_matrix(
            word_index, embedding_dim=args.embedding_dim,
            max_num_words=args.max_num_words
        )
        model = build_DNN(
            embedding_matrix, topic_num=args.topic_num,
            max_sequence_length=args.max_seq_len,
            model_type=args.model,
        )

        # train and save
        callbacks = set_callback(model_name)
        model.fit(X_train, y_train, epochs=10,
                  batch_size=128, callbacks=callbacks,
                  validation_split=0.2)
        # test
        results = model_evaluation(X_test, y_test,
                                   model=model, doc2seq=doc2seq)

    save_file(results, 'classifications', model_name)

    # model testing
    print('done...')
