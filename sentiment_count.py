import csv
import matplotlib.pyplot as plt
import numpy as np

train_file_path = './data/train3.csv'

def counts():
    count = {'positive':0, 'neutral':0, 'negative':0}
    with open(train_file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            count[row[1]] += 1
    return count

def main():
    count = counts()

    plt.rcdefaults()
    fig, ax = plt.subplots()

    labels = count.keys()
    sentiment = list(count.values())
    print(count)
    y_pos = np.arange(len(labels))

    ax.barh(y_pos, sentiment, color=(0.2, 0.4, 0.6, 0.6))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_title('Training Data Sentiment')
    ax.set_xlabel('# of Records')

    plt.show()

if __name__ == '__main__':
    main()