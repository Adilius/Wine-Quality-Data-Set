import pandas as pd
import numpy as np
import seaborn as sns #Visualization
from scipy import stats
from rich.console import Console
import time
import matplotlib.pyplot as plt

#Creates and prints a roc curve
def plot_roc_curve(false_positive_rate, true_positive_rate, title):
    plt.subplots(1, figsize=(10,10))
    plt.title(title)
    plt.plot(false_positive_rate, true_positive_rate)
    plt.plot([0, 1], ls="--")   #Straight line
    plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7") #Straight line
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.draw()  #Create plot

#Preprocessing
def preprocessing(df):

    #Data reduction
    """" Removes duplicate rows """
    df.drop_duplicates(inplace=True)

    #Data cleansing
    """ Remove outliers that are more than 3 std from mean """
    df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    #Data transformation
    """ Normalizes each column into a range between 0 and 1 """
    df = (df-df.min())/(df.max()-df.min())

    return df

#Run Naive Bayes on dataframe
def naive_bayes(df):

    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]
    print(x)
    print(y)


console = Console()

with console.status('[bold green]Preprocessing data...') as status:
    df = pd.read_csv('winequality-white.csv', sep=';')
    df.columns = [x.strip().replace(' ','_') for x in df.columns]
    # print(df)
    # print(df.describe())
    # sns.boxplot(x=df['fixed_acidity'])
    # plt.show()

    df = preprocessing(df)

    # print(df)
    # print(df.describe())
    # sns.boxplot(x=df['fixed_acidity'])
    # plt.show()





with console.status('[bold green]Running Naive Bayes...') as status:
    naive_bayes(df)
    #time.sleep(3)