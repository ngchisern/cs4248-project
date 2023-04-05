import pandas as pd
import numpy as np

from nltk import word_tokenize
from nltk.corpus import stopwords

from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,  Embedding, LSTM, Bidirectional, Layer
from tensorflow.keras.utils import to_categorical
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from gensim.models import KeyedVectors 

import matplotlib.pyplot as plt


def load_data(train_file, test_file, column_name):
    train_df = pd.read_csv(train_file, names=column_name)
    test_df = pd.read_csv(test_file, names=column_name)
    return train_df, test_df


nltk_stopwords = list(stopwords.words('english'))


def tokenize(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in nltk_stopwords]

    return " ".join(tokens)


def word2vec(word):
    try:
        return w2v_model.key_to_index[word]
    except KeyError:
        return 0


def token2word(token):
    return w2v_model.index2word[token]


def preprocess_data(train_df, test_df):
    import time
    start = time.time()
    minute = 0

    for df in [train_df, test_df]:
        for i in range(df.shape[0]):
            if time.time() - start >= minute * 60:
                print(str(minute) + " minutes: " + str(i) + " iterations done.")
                minute += 1
            df.at[i, "text"] = tokenize(df.at[i, "text"])

    return train_df, test_df


def get_train_test_data(x_train, y_train, x_test, y_test, mode, maxlen, n_unique_words):
    # word2vec
    sequences = [[word2vec(word) for word in text.split()] for text in x_train]
    x_train = pad_sequences(sequences, maxlen=maxlen)
    y_train = to_categorical(y_train - 1, num_classes=4)

    if mode == 'valid':
        return train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    sequences = [[word2vec(word) for word in text.split()] for text in x_test]
    x_test = pad_sequences(sequences, maxlen=maxlen)
    y_test = to_categorical(y_test - 1, num_classes=4)
    return x_train, x_test, y_train, y_test


# Attention layer
class AttentionLayer(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(AttentionLayer, self).__init__()

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a

        if self.return_sequences:
            return output
        return K.sum(output, axis=1)


# config
mode = 'test'
maxlen = 200
n_unique_words = 20000

train_df, test_df = load_data("data/fulltrain.csv", "data/balancedtest.csv", ["label", "text"])
# train_df, test_df = preprocess_data(train_df, test_df)

print('Loading word2vec model...')
w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
print('Finished loading word2vec model...')

# Retrieve the weights from the model. This is used for initializing the weights
# in a Keras Embedding layer later
w2v_weights = w2v_model.vectors
vocab_size, embedding_size = w2v_weights.shape


x_train, y_train = train_df["text"].values, train_df["label"].values
x_test, y_test = test_df["text"].values, test_df["label"].values

X_train, X_test, y_train, y_test = get_train_test_data(x_train, y_train, x_test, y_test, mode, maxlen, n_unique_words)

y_train = np.array(y_train)
y_test = np.array(y_test)

model = Sequential()
model.add(Embedding(vocab_size, embedding_size, weights=[w2v_weights], input_length=maxlen, mask_zero=True, trainable=False))
model.add(Bidirectional(LSTM(64, dropout=0.5, recurrent_dropout=0.2, return_sequences=True)))
model.add(AttentionLayer(return_sequences=False))
model.add(Dense(4, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()

model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))
test_pred = model.predict(X_test)
test_pred = np.argmax(test_pred, axis=1)
test_label = np.argmax(y_test, axis=1)

confusion_matrix = confusion_matrix(test_label, test_pred, labels=[0, 1, 2, 3])
print(classification_report(test_label, test_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=["Satire", "Hoax", "Propaganda", "Reliable News"])
disp.plot()
plt.show()
