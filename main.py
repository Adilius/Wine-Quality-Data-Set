import pandas as pd
import numpy as np
import seaborn as sns #Visualization boxplot
from scipy import stats
from rich.console import Console
import time
import matplotlib.pyplot as plt #Plot tool
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
    accuracy = str(metrics.accuracy_score(y_test, y_pred))[:5]
    print('Accuracy KNN:' , accuracy)

    #Calculate recall
    recall = str(metrics.recall_score(y_test, y_pred, average='weighted'))[:5]
    print('Recall KNN:' , recall)

    #Calculate F1-score
    knn_f1_score = str(metrics.f1_score(y_test, y_pred, average='weighted'))[:5]
    print('F1-score KNN:' , knn_f1_score)

#Run KNN on dataframe
def KNN(df, k=17, verbose=True):

    #Split into features and label
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1:]

    #Split into training data set and testing data set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    #Create KNN-model
    knn_model = KNeighborsClassifier(n_neighbors=k, weights='distance', algorithm='brute')

    #Train model
    knn_model.fit(x_train, y_train.values.ravel())

    #Run prediction
    y_pred = knn_model.predict(x_test)

    #Create confusion matrix
    cm = metrics.confusion_matrix(y_test, y_pred)

    #Print confusion matrix
    if verbose:
        print('Confusion matrix KNN:')
        print(cm)

    #Calculate accuracy
    accuracy = str(metrics.accuracy_score(y_test, y_pred))[:5]
    if verbose:
        print('Accuracy KNN:' , accuracy)

    #Calculate recall
    recall = str(metrics.recall_score(y_test, y_pred, average='weighted'))[:5]
    if verbose:
        print('Recall KNN:' , recall)

    #Calculate F1-score
    knn_f1_score = str(metrics.f1_score(y_test, y_pred, average='weighted'))[:5]
    if verbose:
        print('F1-score KNN:' , knn_f1_score)

    return accuracy

#Finds optimal k for KNN
def find_optimal_k(KNN, df, max_k=100):

    acc_list = []
    for k in range(1,max_k):
        acc_list.append(float(KNN(df, k=k, verbose=False)))
    
    optimal_k = acc_list.index(max(acc_list))+1
    print('Optimal k:', optimal_k)

    return optimal_k

console = Console()

with console.status('[bold green]Preprocessing data...') as status:
    time.sleep(1)
    file_name = 'winequality-red.csv'
    print('Data set:',file_name)
    df = pd.read_csv(file_name, sep=';')
    df.columns = [x.strip().replace(' ','_') for x in df.columns]

    #Preprocess data set for naive bayes, normalization not needed
    df_nb = preprocess(copy.deepcopy(df), rm_dup=True, rm_out=True, norm=False)
    df_knn = preprocess(copy.deepcopy(df), rm_dup=True, rm_out=True, norm=True)

with console.status('[bold green]Running Naive Bayes...') as status:
    time.sleep(1)
    nb_start = time.time()
    naive_bayes(df_nb)
    nb_end = time.time()
    print("Naive bayes took:", round(nb_end - nb_start,2),"seconds.")

k = 0
with console.status('[bold green]Calculating optimal k...') as status:
    k = find_optimal_k(KNN, df_knn, max_k=100)

with console.status('[bold green]Running KNN...') as status:
    time.sleep(1)
    knn_start = time.time()
    KNN(df_knn, k=k)
    knn_end = time.time()
    print("KNN took:", round(knn_end - knn_start,2),"seconds.")
