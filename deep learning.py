import pandas as pd
from pandas import DataFrame
import requests
import os
import re
import spacy
from gensim.models.phrases import Phrases, Phraser
from time import time
import multiprocessing
from gensim.models import Word2Vec
import bokeh.plotting as bp
from bokeh.plotting import figure, show, output_notebook
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.models import load_model
from keras import layers
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, Input, Embedding, LSTM, SimpleRNN, Bidirectional
from keras.layers.merge import concatenate
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.callbacks import Callback
from sklearn.manifold import TSNE
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from wordcloud import WordCloud
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import pickle

# model = Word2Vec.load('dv_50.model')
# print(model.most_similar(positive=["abuse"]))
# print(model.doesnt_match("breakfast cereal dinner lunch".split()))


# data = pd.read_excel('DV_dataset.xlsx', sheet_name='Sheet1')
# data['class'][data['class'] == 'fr'] = 1
# data['class'][data['class'] == 'ps'] = 1
# data['class'][data['class'] == 'aw'] = 0
# data['class'][data['class'] == 'em'] = 0
# data['class'][data['class'] == 'gen'] = 0
# print(data['post_id'])
# DataFrame(data).to_excel('DV_dataset.xlsx', sheet_name='Sheet1', index=False, header=True)


df_clean = pd.read_excel('DV_dataset.xlsx', sheet_name='Sheet1')
t = time()
tokenizer = RegexpTokenizer(r'\w+')
df_clean['clean'] = df_clean['post'].astype('str')
df_clean.dtypes
df_clean["tokens"] = df_clean["clean"].apply(tokenizer.tokenize)
# delete Stop Words
print('Time to tokenize everything: {} mins'.format(round((time() - t) / 60, 2)))
df_clean.head()
# print(df_clean)

# WORD2VEC()
# cores = multiprocessing.cpu_count()  # Count the number of cores in a computer, important for a parameter of the model
# w2v_model = Word2Vec(min_count=20,
#                      window=2,
#                      size=50,
#                      sample=6e-5,
#                      alpha=0.03,
#                      min_alpha=0.0007,
#                      negative=20,
#                      workers=cores - 1)

# BUILD_VOCAB()
# t = time()
# w2v_model.build_vocab(df_clean["tokens"], progress_per=1000)
# print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# TRAIN()
# w2v_model.train(df_clean["tokens"], total_examples=w2v_model.corpus_count, epochs=10000, report_delay=1)
# print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
# w2v_model.save('dv_50.model')
w2v_model = Word2Vec.load('dv_50.model')


# First defining the X (input), and the y (output)
y = df_clean['label'].values
X = np.array(df_clean["tokens"])

# And here is the train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# use WordCloud to picture the 100 most important words of the dictionary we had
# vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
# matrix = vectorizer.fit_transform([x for x in X_train])
# tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))

def plot_word_cloud(terms):
    text = terms.index
    text = ' '.join(list(text))
    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure(figsize=(25, 25))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
# tfidf2 = pd.DataFrame(columns=['tfidf']).from_dict(dict(tfidf), orient='index')
# tfidf2.columns = ['tfidf']
# plot_word_cloud(tfidf2.sort_values(by=['tfidf'], ascending=True).head(100))


# all_words = [word for tokens in X for word in tokens]
# all_sentence_lengths = [len(tokens) for tokens in X]
# ALL_VOCAB = sorted(list(set(all_words)))
# print("%s words total, with a vocabulary size of %s" % (len(all_words), len(ALL_VOCAB)))
# print("Max sentence length is %s" % max(all_sentence_lengths))

# ####################### CHANGE THE PARAMETERS HERE #####################################
EMBEDDING_DIM = 50  # how big is each word vector
MAX_VOCAB_SIZE = 11085  # how many unique words to use (i.e num rows in embedding vector)
MAX_SEQUENCE_LENGTH = 500   # max number of words in a comment to use   4507


tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, char_level=False)
tokenizer.fit_on_texts(df_clean["post"].tolist())
training_sequences = tokenizer.texts_to_sequences(X_train.tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))

train_embedding_weights = np.zeros((len(train_word_index)+1, EMBEDDING_DIM))
for word, index in train_word_index.items():
    train_embedding_weights[index, :] = w2v_model[word] if word in w2v_model else np.random.rand(EMBEDDING_DIM)
print(train_embedding_weights.shape)


######################## TRAIN AND TEST SET #################################
train_data = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)    #截断函数，使每个输入为长度相同的矩阵
test_sequences = tokenizer.texts_to_sequences(X_test.tolist())
test_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        return


metrics = Metrics()

# CNN
# Define the CNN
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words,
                                embedding_dim,
                                weights=[embeddings],
                                input_length=max_sequence_length,
                                trainable=trainable)

    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3, 4, 5]

    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate([convs[0], convs[1], convs[2]], axis=1)

    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)

    if extra_conv == True:
        x = Dropout(0.5)(l_merge)
    else:
        # Original Yoon Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # Finally, we feed the output into a Sigmoid layer.
    # The reason why sigmoid is used is because we are trying to achieve a binary classification(1,0)
    # for each of the 6 labels, and the sigmoid function will squash the output between the bounds of 0 and 1.
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['acc'])
    model.summary()
    return model


# cnn_model = ConvNet(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM, False)
#
# cnn_history = cnn_model.fit(train_data, y_train, epochs=20, batch_size=64, validation_data=(test_data, y_test), callbacks=[metrics])
#
# loss, accuracy = cnn_model.evaluate(train_data, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = cnn_model.evaluate(test_data, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# print("Precision:  {:.4f}".format(metrics.val_precisions[-1]))
# print("Recall:  {:.4f}".format(metrics.val_recalls[-1]))
# print("F1-score:  {:.4f}".format(metrics.val_f1s[-1]))


#RNN
def RNN(embeddings, max_sequence_length, num_words, embedding_dim):
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embeddings],
                        input_length=max_sequence_length,
                        trainable=False))
    model.add(SimpleRNN(max_sequence_length, dropout=0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.summary()
    return model


rnn_model = RNN(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM)   #MAX_SEQUENCE_LENGTH=500
rnn_history = rnn_model.fit(train_data, y_train, epochs=20, batch_size=64, validation_data=(test_data, y_test), callbacks=[metrics])

loss, accuracy = rnn_model.evaluate(train_data, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = rnn_model.evaluate(test_data, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Precision:  {:.4f}".format(metrics.val_precisions[-1]))
print("Recall:  {:.4f}".format(metrics.val_recalls[-1]))
print("F1-score:  {:.4f}".format(metrics.val_f1s[-1]))


# LSTM
def lstm(embeddings, max_sequence_length, num_words, embedding_dim):
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embeddings],
                        input_length=max_sequence_length,
                        trainable=False))
    model.add(LSTM(max_sequence_length, dropout=0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.summary()
    return model


# lstm_model = lstm(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM)   #MAX_SEQUENCE_LENGTH=500
# lstm_history = lstm_model.fit(train_data, y_train, epochs=20, batch_size=64, validation_data=(test_data, y_test), callbacks=[metrics])
#
# loss, accuracy = lstm_model.evaluate(train_data, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = lstm_model.evaluate(test_data, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# print("Precision:  {:.4f}".format(metrics.val_precisions[-1]))
# print("Recall:  {:.4f}".format(metrics.val_recalls[-1]))
# print("F1-score:  {:.4f}".format(metrics.val_f1s[-1]))


# Bi-LSTM+Attention
def bilstm(embeddings, max_sequence_length, num_words, embedding_dim):
    model = Sequential()
    model.add(Embedding(num_words,
                        embedding_dim,
                        weights=[embeddings],
                        input_length=max_sequence_length,
                        trainable=False))
    model.add(Bidirectional(LSTM(max_sequence_length, dropout=0.5, return_sequences=True)))
    model.add(SeqWeightedAttention())   #不能用SeqSelfAttention，因为这样输出还是三维的，现在想输入为2维的
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['acc'])
    model.summary()
    return model


# bilstm_model = bilstm(train_embedding_weights, MAX_SEQUENCE_LENGTH, len(train_word_index)+1, EMBEDDING_DIM)   #MAX_SEQUENCE_LENGTH=500
# bilstm_history = bilstm_model.fit(train_data, y_train, epochs=20, batch_size=64, validation_data=(test_data, y_test), callbacks=[metrics])
#
# loss, accuracy = bilstm_model.evaluate(train_data, y_train, verbose=False)
# print("Training Accuracy: {:.4f}".format(accuracy))
# loss, accuracy = bilstm_model.evaluate(test_data, y_test, verbose=False)
# print("Testing Accuracy:  {:.4f}".format(accuracy))
# print("Precision:  {:.4f}".format(metrics.val_precisions[-1]))
# print("Recall:  {:.4f}".format(metrics.val_recalls[-1]))
# print("F1-score:  {:.4f}".format(metrics.val_f1s[-1]))

# Save and load the model
rnn_model.save('w2v50_rnn.h5')
del rnn_model
print("Saved model to disk!")
# model = load_model('w2v50_rnn.h5')

# Save and load the history
with open('rnn_trainHistoryDict.txt', 'wb') as file_pi:
    pickle.dump(rnn_history.history, file_pi)
print("Saved history to disk!")
with open('rnn_trainHistoryDict.txt', 'rb') as file_pi:
    rnn_history = pickle.load(file_pi)


def plot_history(history):
    acc = history['acc']
    val_acc = history['val_acc']
    loss = history['loss']
    val_loss = history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('RNN Training and validation accuracy')
    plt.xlim((0, 21))
    plt.ylim((0.3, 1))
    my_x_ticks = np.arange(0, 21, 2)
    my_y_ticks = np.arange(0.3, 1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('RNN Training and validation loss')
    plt.xlim((0, 21))
    my_x_ticks = np.arange(0, 21, 2)
    plt.xticks(my_x_ticks)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# plot_history(rnn_history)


def plot_comparison():
    with open('cnn_trainHistoryDict.txt', 'rb') as file_pi:
        cnn_history = pickle.load(file_pi)
    with open('rnn_trainHistoryDict.txt', 'rb') as file_pi:
        rnn_history = pickle.load(file_pi)
    with open('lstm_trainHistoryDict.txt', 'rb') as file_pi:
        lstm_history = pickle.load(file_pi)
    with open('bilstm_trainHistoryDict.txt', 'rb') as file_pi:
        bilstm_history = pickle.load(file_pi)
    cnn_acc = cnn_history['val_acc']
    cnn_loss = cnn_history['val_loss']
    rnn_acc = rnn_history['val_acc']
    rnn_loss = rnn_history['val_loss']
    lstm_acc = lstm_history['val_acc']
    lstm_loss = lstm_history['val_loss']
    bilstm_acc = bilstm_history['val_acc']
    bilstm_loss = bilstm_history['val_loss']

    x = range(1, 21)
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.plot(x, cnn_acc, 'g', label='CNN Validation acc')
    plt.plot(x, rnn_acc, 'y', label='RNN Validation acc')
    plt.plot(x, lstm_acc, 'b', label='LSTM Validation acc')
    plt.plot(x, bilstm_acc, 'r', label='Bi-LSTM Validation acc')
    plt.title('Comparison of Validation Accuracy')
    plt.xlim((0, 21))
    plt.ylim((0.3, 1))
    my_x_ticks = np.arange(0, 21, 2)
    my_y_ticks = np.arange(0.3, 1, 0.1)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, cnn_loss, 'g', label='CNN Validation loss')
    plt.plot(x, rnn_loss, 'y', label='RNN Validation loss')
    plt.plot(x, lstm_loss, 'b', label='LSTM Validation loss')
    plt.plot(x, bilstm_loss, 'r', label='Bi-LSTM Validation loss')
    plt.title('Comparison of Validation Loss')
    plt.xlim((0, 21))
    my_x_ticks = np.arange(0, 21, 2)
    plt.xticks(my_x_ticks)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# plot_comparison()