import time
import os
import re
import string
import csv
import pickle
import argparse
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


parser = argparse.ArgumentParser(description='Generate text classifier')
parser.add_argument('--build', help='build a model for classification and store it in ./predicition', action='store_true')
parser.add_argument('--classify', help='If a model exists use existing model to classify test3.csv', action='store_true')
args = parser.parse_args()

NFOLDS = 5 
STOPWORDS = set()

train_file_path = './data/train3.csv'
test_file_path = './data/test3.csv'
stop_words_file_path = 'stop_words.txt'
model_file_path = './model/pickle_model.pkl'
prediction_file_path = './prediction/debug_prediction.csv'
prediction_result_file_path = './prediction/prediction.csv'

def load_stop_words(filepath):
    with open(filepath, 'r') as file:
        for line in file:
            STOPWORDS.add(line.replace('\n', ''))

def clean_text(text, remove_stop_words=False, stem=False):
    text = text.lower()
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(replace_punctuation)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('[\n\r]', '', text)

    if remove_stop_words == True or stem == True:
        text = text.split()
        new_text = []
        stemmer = PorterStemmer()       
        for word in text:
            if stem == True:
                word = stemmer.stem(word)
            if remove_stop_words:
                if word not in STOPWORDS:
                    new_text.append(word)
            else:
                new_text.append(word)

        text = ' '.join(new_text)

    return text

def load_train(filepath):
    df = pd.DataFrame(columns=['text', 'sent'])
    text = []
    sent = []

    with open(filepath, 'r') as train:
        reader = csv.reader(train, delimiter = ',')
        next(reader, None)
        for row in reader:
            review = clean_text(row[0], remove_stop_words=True, stem=True)

            text.append(review)
            sent.append(row[1])

    df['text'] = text
    df['sent'] = sent
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def load_test(filepath):
    df = pd.DataFrame(columns=['ID', 'text', 'raw_text'])
    ID = []
    text = []
    raw_text = []

    with open(filepath, 'r') as test:
        reader = csv.reader(test, delimiter=',')
        next(reader, None)
        for row in reader:
            raw_review = row[1]
            review = clean_text(row[1], remove_stop_words=True, stem=True)
            IDs = row[0]

            raw_text.append(raw_review)
            text.append(review)
            ID.append(IDs)
    
    df['ID'] = ID
    df['text'] = text
    df['raw_text'] = raw_text
    return df

def build_classifier(dataFrame):
    X = dataFrame.iloc[:, 0].values
    Y = dataFrame.iloc[:, 1].values

    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(solver='sag', C=1.0))
    ])

    i = 0
    kf = KFold(n_splits=NFOLDS, shuffle=True)
    for train_index, test_index in kf.split(X):
        i += 1
        print("\n******************* Running fold: %d/%d *******************\n" % (i, NFOLDS))

        text_clf.fit(X[train_index], Y[train_index])
        
        accuracy = text_clf.score(X[train_index], Y[train_index])
        val_accuracy = text_clf.score(X[test_index], Y[test_index])
        y_pred = text_clf.predict(X[test_index])
        cm = confusion_matrix(Y[test_index], y_pred)
        print("accuracy: %f val_accuracy: %f\n" % (accuracy, val_accuracy))
        print(cm)
    
    return text_clf

def main():
    start = time.time()

    if args.build:
        load_stop_words(stop_words_file_path)
        train_df = load_train(train_file_path)
        clf = build_classifier(train_df)

        with open(model_file_path, 'wb') as file:
            pickle.dump(clf, file)
    
    if args.classify:
        model = None
        with open(model_file_path, 'rb') as file:
            model = pickle.load(file)

        test_df = load_test(test_file_path)
        ID = test_df.iloc[:, 0]
        X_test = test_df.iloc[:, 1]
        raw_text = test_df.iloc[:, 2]
        y_test_pred = model.predict(X_test)
        
        df = pd.DataFrame({'ID': ID, 'text': raw_text, 'sent': y_test_pred})
        df_result = pd.DataFrame(columns=['REVIEW-ID', 'CLASS'])
        df_result['REVIEW-ID'] = ID
        df_result['CLASS'] = y_test_pred

        df.to_csv(path_or_buf=prediction_file_path, index=False)
        df_result.to_csv(path_or_buf=prediction_result_file_path, index=False)

    end = time.time()
    print('Model completed in %d seconds' % (end - start))
    
if __name__ == '__main__':
    main()