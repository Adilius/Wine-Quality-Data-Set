import pandas as pd
import numpy as np
from rich.console import Console
import time
import matplotlib.pyplot as plt

#Preprocessing
def preprocessing(df):

    #Data reduction
    df.dropna(inplace=True)

    #Data cleansing

    #Data transformation

    return df

#Plot ROC curve
def plot_roc_curve():
    pass




console = Console()

with console.status('[bold green]Preprocessing data...') as status:
    df = pd.read_csv('winequality-white.csv')
    df = preprocessing(df)
    print(df)
    print(df['volatile acidity'].describe())

#with console.status('[bold green]Training data...') as status:
#    time.sleep(3)