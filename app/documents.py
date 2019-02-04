
from spark_data import make_rdd
from helper import (load_category_topic_mapping,
                        get_path, load_topic_mapping)

import os
import pickle
from pyspark import SparkContext
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    sc = SparkContext("local[*]", "temp")

    topic_mapping = load_category_topic_mapping()
    group_rdd = make_rdd(sc, 'group') \
        .map(lambda grp: (topic_mapping[grp.category], grp.description))\
        .filter(lambda x: all(x)) \
        .cache()

    counts = group_rdd \
        .map(lambda x: (x[0], 1))  \
        .reduceByKey(lambda x, y: x + y).collect()

    topics = sorted(counts, reverse=True, key=lambda x: x[1])

    # # train test split
    for topicN in [5, 10, 15, 20]:
        selected = [t[0] for t in topics[:topicN]]

        filtered = group_rdd \
            .filter(lambda x: x[0] in selected)
        descriptions = filtered.map(lambda x: x[1]).collect()
        labels = filtered.map(lambda x: x[0]).collect()

        # mapping topic names to ids
        mapping = load_topic_mapping()
        labels = [mapping[l] for l in labels]

        # train test split
        X_train, X_test, y_train, y_test = train_test_split(
            descriptions, labels, test_size=0.2, random_state=42
        )
        train = (X_train, y_train)
        test = (X_test, y_test)
        # file name
        data_path = get_path('data', 'documents')
        train_file = os.path.join(data_path, f'train_{topicN:02d}')
        test_file = os.path.join(data_path, f'test_{topicN:02d}')
        # save
        pickle.dump(train, open(train_file, 'wb'))
        pickle.dump(test, open(test_file, 'wb'))
