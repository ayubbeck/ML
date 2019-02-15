import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# plot all columns in histogram format
def histogram(df):
    fig = plt.figure(figsize=(20,15))
    cols = 5
    rows = math.ceil(float(df.shape[1]) / cols)
    for i, column in enumerate(df.columns):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.set_title(column)
        if df.dtypes[column] == np.object:
            df[column].value_counts().plot(kind="bar", axes=ax)
        else:
            df[column].hist(axes=ax)
            plt.xticks(rotation="vertical")
    plt.subplots_adjust(hspace=0.7, wspace=0.2)
    plt.show()

def heatmap(df):
    plt.subplots(figsize=(20,20))
    # print(df.astype('float64').corr())
    sns.heatmap(df.corr(), square=True)
    plt.show()
