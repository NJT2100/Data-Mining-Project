import time
import re
import string
import csv
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import ComplementNB
from sklearn import svm
from sklearn.utils.class_weight import compute_sample_weight
from nltk.stem import WordNetLemmatizer, PorterStemmer

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

def clean_text(text, remove_stop_words=False):
    text = text.lower()
    replace_punctuation = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text = text.translate(replace_punctuation)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('[\n\r]', '', text)

    if remove_stop_words == True:
        text = text.split()
        new_text = []
        stemmer = PorterStemmer()
        
        for word in text:
            if word not in STOPWORDS:
                new_text.append(stemmer.stem(word))

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

def main():
    start = time.time()

    load_stop_words('stop_words.txt')
    train_df = load_train(train_file_path)
    X = train_df.iloc[:, 0].values
    Y = train_df.iloc[:, 1].values

    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', ComplementNB(alpha=0.5))
    ])

    i = 0
    kf = KFold(n_splits=NFOLDS, shuffle=True)
    for train_index, test_index in kf.split(X):
        i += 1
        print("\n************************************* Running fold: %d/%d *****************************************\n" % (i, NFOLDS))

        text_clf.fit(X[train_index], Y[train_index])
        
        accuracy = text_clf.score(X[train_index], Y[train_index])
        val_accuracy = text_clf.score(X[test_index], Y[test_index])
        y_pred = text_clf.predict(X[test_index])
        cm = confusion_matrix(Y[test_index], y_pred)
        print("accuracy: %f val_accuracy: %f\n" % (accuracy, val_accuracy))
        print(cm)

    end = time.time()
    print('Model completed in %d seconds' % (end - start))
    
if __name__ == '__main__':
    main()