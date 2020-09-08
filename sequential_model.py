import re
import string
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Embedding, GlobalAveragePooling1D
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.feature_extraction import text
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

EPOCHS=10
MAX_WORDS = 2500
SEQ_LEN = 512
EMBEDDING_SIZE = 32
BATCH_SIZE = 16
NFOLDS = 5
CLASS = 'class'
LABELS = ['positive', 'negative', 'neutral']
STOPWORDS = text.ENGLISH_STOP_WORDS

train_file_path = './data/train3.csv'
test_file_path = './data/test3.csv'

def load_stop_words(filepath):
    stopwords = set()
    with open('stop_words.txt', 'r') as file:
        for line in file:
            stopwords.add(line.replace('\n', ''))

    return stopwords


def clean_text(text, remove_stop_words):
    text = text.lower()
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(replace_punctuation)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('[\n\r]', '', text)

    if remove_stop_words == True:
        text = text.split()
        new_text = []
        lemmatizer = WordNetLemmatizer()
        
        for word in text:
            if word not in STOPWORDS:
                new_text.append(lemmatizer.lemmatize(word))

        text = ' '.join(new_text)

    return text

def load_train(filepath):
    df = pd.DataFrame(columns=['text', 'sent'])
    text = []
    sent = []

    with open(train_file_path, 'r') as train:
        reader = csv.reader(train, delimiter = ',')
        for row in reader:
            review = clean_text(row[0], True)

            text.append(review)
            if row[1] == 'neutral':
                sent.append(0)
            elif row[1] == 'positive':
                sent.append(1)
            else:
                sent.append(-1)

    df['text'] = text
    df['sent'] = sent
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def convert_text_to_index(dataframe):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, oov_token='<UNK>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(dataframe['text'])

    #convert text data to numerical indexes
    train_seq = tokenizer.texts_to_sequences(dataframe['text'])

    #pad data up to SEQ_LEN (note that we truncate if there are more than SEQ_LEN tokens)
    train_seq = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=SEQ_LEN, padding='post')

    return train_seq

def create_model():
    model = keras.Sequential()

    model.add(Embedding(MAX_WORDS, EMBEDDING_SIZE))
    model.add(Dropout(0.5))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(1, activation='softmax'))

    model.summary()

    #mean_squared_error, huber_loss, mean_absolute_error, mean_squared_logarithmic_error
    model.compile(optimizer='adagrad', loss='mean_squared_error', metrics=['accuracy'])

    return model

def main():
    STOPWORDS = load_stop_words('stop_words.txt')
    train_df = load_train(train_file_path)

    print(train_df)
 
    #define callback to stop building model when validation accuracy decreases (epochs stop building)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='max')
    callbacks=[early_stop]

    #convert the text to indexs for kera
    train_seq = convert_text_to_index(train_df)

    i = 0
    kf = KFold(n_splits=NFOLDS, shuffle=True)
    for train_index, test_index in kf.split(train_seq):
        i += 1
        print("\n************************************* Running fold: %d/%d *****************************************\n" % (i, NFOLDS))
        x_train, x_test = train_seq[train_index], train_seq[test_index]
        y_train, y_test = train_df['sent'][train_index], train_df['sent'][test_index]
        model = create_model()
        history = model.fit(x_train, y_train,
                                batch_size=BATCH_SIZE,
                                epochs=EPOCHS,
                                validation_split=0.0,
                                validation_data=(x_test, y_test),
                                callbacks=callbacks)

        print('Model evaluation ',model.evaluate(x_test, y_test))

if __name__ == '__main__':
    main()