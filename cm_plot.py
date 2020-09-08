import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def main():
    cm = [[1861, 116, 333],
        [279, 252, 675],
        [142, 124, 7418]]
    df_cm = pd.DataFrame(cm, index = ['Neg', 'Neu', 'Pos'], columns= ['Neg', 'Neu', 'Pos'])

    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={'size': 10}, cmap=sn.cm.rocket_r, fmt='g')
    
    plt.title('Sentiment Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()