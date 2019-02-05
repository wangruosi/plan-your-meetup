"""
Deep learning models
@Ruosi Wang ruosiwang.psy@gmail.com
"""
from helper import get_path, load_glove

import os
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, GlobalMaxPooling1D, Dropout, CuDNNGRU
from keras.layers import Conv1D, MaxPooling1D, Embedding, Flatten
from keras.models import Sequential, Model
from keras.initializers import Constant


models_path = get_path('results', 'models')


def set_callback(model_name):
    filepath = os.path.join(models_path, model_name)
    callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                 ModelCheckpoint(filepath=filepath,
                                 monitor='val_loss',
                                 save_best_only=True)]
    return callbacks


def build_DNN(embedding_matrix, max_sequence_length=500,
              topic_num=5, model_type=None):
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_sequence_length,
                                trainable=False)
    model = Sequential()
    model.add(embedding_layer)

    if model_type == 'MLP':
        model = _add_MLP(model)
    if model_type == 'GRU':
        model = _add_GRU(model)
    if model_type == 'CNN':
        model = _add_CNN(model)
    if model_type == "CNN_GRU":
        model = _add_CNN_GRU(model)

    model.add(Dense(topic_num, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['acc'])

    return model


def _add_MLP(model, layer_num=2):
    model.add(Flatten())

    for _ in range(layer_num):
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.2))

    return model


def _add_CNN(model, layer_num=2):
    for i in range(layer_num):
        model.add(Conv1D(128, 5, activation='relu'))
        if i == layer_num - 1:
            model.add(GlobalMaxPooling1D())
        else:
            model.add(MaxPooling1D(5))
        model.add(Dropout(0.2))
    return model


def _add_GRU(model, layer_num=1):
    for _ in range(layer_num):
        model.add(CuDNNGRU(100))
        model.add(Dropout(0.2))
    return model


def _add_CNN_GRU(model, layer_num=[1, 1]):
    cnn_lN, gru_lN = layer_num
    for _ in range(layer_num[0]):
        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(5))
        model.add(Dropout(0.2))

    for _ in range(layer_num[1]):
        model.add(CuDNNGRU(100))
        model.add(Dropout(0.2))
    return model


# def tokenizer_transform(X_train, X_test,
#                         max_num_words=20000,
#                         max_sequence_length=500):
#     # tokenize group descriptions
#     tokenizer = Tokenizer(num_words=max_num_words)
#     tokenizer.fit_on_texts(X_train)
#     X_train = pad_sequences(
#         tokenizer.texts_to_sequences(X_train),
#         maxlen=max_sequence_length
#     )
#     print("preprocessing done...")
#     return X_train, X_test, tokenizer

def tokenizer_transform(X_train, max_num_words=20000,
                        max_sequence_length=500):
    # tokenize group descriptions
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(X_train)

    def doc2seq(doc):
        seq = tokenizer.texts_to_sequences(doc)
        return pad_sequences(seq, max_sequence_length)

    X_train_seq = doc2seq(X_train)
    print("preprocessing done...")
    return X_train_seq, tokenizer.word_index, doc2seq


def get_embedding_matrix(word_index, embedding_dim=50,
                         max_num_words=20000):
    """
    get embedding matrix
    """
    num_words = min(len(word_index), max_num_words) + 1
    embedding_matrix = np.zeros((num_words, embedding_dim))
    embedding_index = load_glove(embedding_dim=embedding_dim)

    for word, i in word_index.items():
        if i < max_num_words:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


def get_dense_layer_output(model):
    # with a Sequential model
    # file = os.path.join(MODEL_DIR, model_type)
    # model = load_model(file)
    layer_name = model.layers[-2].name
    new_model = Model(inputs=model.input,
                      outputs=model.get_layer(layer_name).output)
    return new_model
