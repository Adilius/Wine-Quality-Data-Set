import pandas as pd
import numpy as np
import seaborn as sns #Visualization
from scipy import stats
from rich.console import Console
import time
import matplotlib.pyplot as plt
import copy #For deep copy

from sklearn.model_selection import train_test_split  # Naive bayes
from sklearn.naive_bayes import GaussianNB  # Naive bayes
from sklearn import preprocessing  # Naive bayes
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.20
RANDOM_STATE = 40

#Preprocessing
def preprocess(df, rm_dup, rm_out, norm):

    #Data reduction
    """" Removes duplicate rows """
    if(rm_dup == True):
        df = df.drop_duplicates()

    #Data cleansing
    """ Remove outliers that are more than 3 std from mean """
    if(rm_out == True):
        df = df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]

    #Data transformation
    """ Normalizes each column into a range between 0 and 1 """
    if(norm == True):
        df.iloc[:, :-1] = df.iloc[:, :-1].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

    return df

#Run Naive Bayes on dataframe
def naive_bayes(df):

    #Split into features and label
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    #Split into training data set and testing data set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    #Create Naive Bayes model
    nb_model = GaussianNB()

    #Train the model
    nb_model.fit(x_train, y_train.values.ravel())

    #Run prediction
    y_pred = nb_model.predict(x_test)

    #Create confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    #Print confusion matrix
    print('Confusion matrix NB:')
    print(cm)

    #Calculate accuracy
    print('Accuracy NB:' , metrics.accuracy_score(y_test, y_pred))

#Run KNN on dataframe
def KNN(df):

    #Split into features and label
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    #Split into training data set and testing data set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    #Create KNN-model
    knn_model = KNeighborsClassifier(n_neighbors=5)

    #Train model
    knn_model.fit(x_train, y_train.values.ravel())

    #Run prediction
    y_pred = knn_model.predict(x_test)

    #Create confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    #Print confusion matrix
    print('Confusion matrix KNN:')
    print(cm)

    #Calculate accuracy
    print('Accuracy KNN:' , metrics.accuracy_score(y_test, y_pred))


console = Console()

with console.status('[bold green]Preprocessing data...') as status:
    time.sleep(1)
    df = pd.read_csv('winequality-red.csv', sep=';')
    df.columns = [x.strip().replace(' ','_') for x in df.columns]
    # print(df)
    print(df.describe())
    #sns.boxplot(x=df['fixed_acidity'])
    
    plt.scatter(df['residual_sugar'], df['chlorides'])
    plt.xlim(0, 20)
    plt.ylim(-10,10)
    plt.show()

    #Preprocess data set for naive bayes, normalization not needed
    df_nb = preprocess(copy.deepcopy(df), rm_dup=True, rm_out=False, norm=False)
    df_knn = preprocess(copy.deepcopy(df), rm_dup=True, rm_out=True, norm=True)
    plt.scatter(df_knn['residual_sugar'], df_knn['chlorides'])
    plt.show()
    print(df_knn.describe())
    #print('df_nb', df_nb['quality'].describe())
    #print('df_knn',df_knn['quality'].describe())


with console.status('[bold green]Running Naive Bayes...') as status:
    time.sleep(1)
    naive_bayes(df_nb)


with console.status('[bold green]Running KNN...') as status:
    time.sleep(1)
    KNN(df_knn)
