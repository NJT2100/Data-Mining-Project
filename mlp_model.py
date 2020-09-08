import re
import string
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import KFold
from sklearn.feature_extraction import text
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

EPOCHS=10
MAX_WORDS = 2500
SEQ_LEN = 512
BATCH_SIZE = 16
NFOLDS = 5
CLASS = 'class'
LABELS = ['positive', 'negative', 'neutral']
STOPWORDS = set()

train_file_path = './data/train3.csv'
test_file_path = './data/test3.csv'

def load_stop_words(filepath):
    with open('stop_words.txt', 'r') as file:
        for line in file:
            STOPWORDS.add(line.replace('\n', ''))

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
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_WORDS, oov_token='<UNK>', 
                                                filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(dataframe['text'])

    #convert text data to numerical indexes
    train_seq = tokenizer.texts_to_sequences(dataframe['text'])

    #pad data up to SEQ_LEN (note that we truncate if there are more than SEQ_LEN tokens)
    train_seq = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=SEQ_LEN, padding='post')

    return train_seq

def main():
    load_stop_words('stop_words.txt')
    train_df = load_train(train_file_path)

    print(train_df)
 
    #convert the text to indexs for model
    X = convert_text_to_index(train_df)
    Y = train_df['sent']
    print(X)

    clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
                batch_size=BATCH_SIZE, solver='sgd', verbose=True,  random_state=21,tol=0.000000001)

    i = 0
    kf = KFold(n_splits=NFOLDS, shuffle=True)
    for train_index, test_index in kf.split(X):
        i += 1
        print("\n************************************* Running fold: %d/%d *****************************************\n" % (i, NFOLDS))

        clf.fit(X[train_index], Y[train_index])
        y_pred = clf.predict(X[test_index])

        acc = accuracy_score(Y[test_index], y_pred)
        cm = confusion_matrix(Y[test_index], y_pred)

        print('Accuracy: %f' % acc)
        print(cm)

if __name__ == '__main__':
    main()