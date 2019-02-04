"""
meetup documents topic classifier
@Ruosi Wang ruosiwang.psy@gmail.com

"""

from helper import get_path
from sklearn_models import build_SKM

import json
import pickle
import argparse

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
def model_evaluation(model, model_name, X_test, y_test):
    # predictions
    y_pred = model.predict(X_test)
    # metrics
    f1 = f1_score(y_pred, y_test, average='weighted')
    acc = accuracy_score(y_pred, y_test)
    conf = confusion_matrix(y_pred, y_test)
    print(f'f1 score: {f1 * 100:.2f}%; accuracy: {acc * 100: .2f}%')
    # save
    file = get_path('results', 'classifications', model_name)
    pickle.dump((f1, acc, conf), open(file, 'wb'))


def save_model(model, model_name):
    file = get_path('results', 'models', model_name)
    pickle.dump(model, open(file, 'wb'))


if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='MNB',
                        help='What is the model?')
    parser.add_argument('--topic_num', type=int, default=5,
                        help='What is the number of topics to classify?')
    parser.add_argument('--embedding_dim', type=int, default=50,
                        help='What is the number of embedding dim?')
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
        # traing pipe line
        skm = build_SKM(model_type=args.model,
                        max_features=20000,
                        selectK=10000)

        # grid search
        clf = GridSearchCV(skm, params, cv=3, n_jobs=-1)
        clf.fit(X_train, y_train)

        # train with best params
        skm.set_params(**clf.best_params_)
        skm.fit(X_train, y_train)

        # test and save
        model_evaluation(skm, model_name, X_test, y_test)
        save_model(skm, model_name)

    # model testing
    print('done...')
