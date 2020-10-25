"""
    Utility helper file
"""
from modules.attention_layer import Attention, AttentionWithContext, Addition
from modules.constants import *

import re
import pathlib

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import pandas as pd

from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.layers import Bidirectional

# ----- KERAS -----
from keras import initializers, optimizers
from keras.models import Sequential
# ----- KERAS -----

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from sklearn.metrics import roc_curve, auc


def get_imdb_reviews(path_to_directory=None):
    """
        Test function fixed into iMDB example
    :param path_to_directory: (str) path to directory of reviews
    :return: (list) list of reviews
    """
    list_of_reviews = []

    for path in pathlib.Path(path_to_directory).iterdir():
        if path.is_file():
            current_file = open(path, "r")
            list_of_reviews.append(current_file.read())
            current_file.close()

    return list_of_reviews


def import_dataset(path_to_file=None):
    """
        Import the dataset based on the path to file URL

    :param path_to_file: (str) path to CSV file
    :return: (pandas.Dataframe) Pandas dataframe containing data
    """
    return pd.read_csv(path_to_file)


def clean_text(text):
    """
        Utility function to help with cleaning up raw-input text. The following is performed:
            - filtering out punctuations
            - transform text to lowercase
            - reduce words to their roots

    :param text: (str) input raw-text to be processed
    :return: (str) processed text
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = re.sub(r'[^\w\s]', '', str(text))
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text


def build_temporal_model(max_len, hidden_units, num_layers, is_attention, is_bidirectional):
    adam = optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)

    model = Sequential()
    model.add(Embedding(MAX_FEATURES, EMBED_SIZE, input_length=max_len))

    # sequence_input = Input(shape=(max_len,), dtype="int32")
    # embedded_sequences = Embedding(max_features, embed_size)(sequence_input)
    #
    # lstm = Bidirectional(LSTM(cell_size, return_sequences=True), name="bi_lstm_0")(embedded_sequences)

    for i in range(num_layers):
        return_sequences = is_attention or (num_layers > 1 and i < num_layers - 1)

        if is_bidirectional:
            model.add(Bidirectional(LSTM(hidden_units, return_sequences=True),
                                    name="bi_lstm_{}".format(i),
                                    merge_mode='concat'))
        else:
            model.add(LSTM(hidden_units, return_sequences=return_sequences, dropout=0.2,
                           kernel_initializer=initializers.glorot_normal(seed=777), bias_initializer='zeros'))

        if is_attention:
            model.add(AttentionWithContext())
            model.add(Addition())

    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=METRICS)

    return model


def build_bidirectional_lstm(X, max_features, embed_size):
    """
        Builder function for the Bidirectional LSTM with user-specification of an Attention Layer.

    :param X: (ndarray) input dataset
    :param max_features: (int) maximum number of features
    :param embed_size: (int) size of embedding
    :return: (tf.keras.Model) LSTM model
    """
    # sequence_input = Input(shape=(max_len,), dtype="int32")
    # embedded_sequences = Embedding(max_features, embed_size)(sequence_input)
    #
    # lstm = Bidirectional(LSTM(cell_size, return_sequences=True), name="bi_lstm_0")(embedded_sequences)
    #
    # # Getting our LSTM outputs
    # (lstm, forward_h, forward_c, backward_h, backward_c) = Bidirectional(LSTM(cell_size,
    #                                                                           return_sequences=True,
    #                                                                           return_state=True),
    #                                                                      name="bi_lstm_1")(lstm)
    #
    # state_h = Concatenate()([forward_h, backward_h])
    # state_c = Concatenate()([forward_c, backward_c])
    # context_vector, attention_weights = Attention(10)(lstm, state_h)
    # dense1 = Dense(20, activation="relu")(context_vector)
    # dropout = Dropout(0.05)(dense1)
    # output = Dense(1, activation="sigmoid")(dropout)
    #
    # return keras.Model(inputs=sequence_input, outputs=output)

    model = Sequential()

    model.add(Embedding(max_features, embed_size, input_length=X.shape[1]))
    model.add(Bidirectional(LSTM(196, dropout=0.2)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)

    return model


def plot_confusion_matrix(labels, predictions):
    """
        Plot the confusion matrix of the model's predictions

    :param labels: (list) ground truth labels
    :param predictions: (list) model predictions
    """
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion matrix (non-normalized))")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")

    plt.show()


def plot_model_metrics(history):
    """
        Plot the following metrics based on model history
            - loss
            - true positive
            - false positive
            - true negative
            - false negative
            - accuracy
            - precision
            - recall
            - area under curve

    :param history: (tf.keras.Model) model that has been fit to data
    """
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mpl.rcParams["figure.figsize"] = (12, 18)

    metrics = [
        "loss",
        "tp", "fp", "tn", "fn",
        "accuracy",
        "precision", "recall",
        "auc",
    ]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(5, 2, n + 1)
        plt.plot(
            history.epoch,
            history.history[metric],
            color=colors[0],
            label="Train",
        )
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[1],
            linestyle="--",
            label="Val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1] * 1.2])
        elif metric == "accuracy":
            plt.ylim([0.4, 1])
        elif metric == "fn":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "fp":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "tn":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "tp":
            plt.ylim([0, plt.ylim()[1]])
        elif metric == "precision":
            plt.ylim([0, 1])
        elif metric == "recall":
            plt.ylim([0.4, 1])
        else:
            plt.ylim([0, 1])

        plt.legend()


def plot_roc_curve(labels, predictions):
    """
        Plot the ROC curve of the model's predictions

    :param labels: (list) ground truth
    :param predictions: (list) model predictions
    """
    mpl.rcParams["figure.figsize"] = (6, 6)

    n_classes = 1

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(labels.ravel(), labels.ravel())
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(labels.ravel(), predictions.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
