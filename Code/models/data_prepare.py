import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

# path = 'E:\PersonalFiles\杂项\Dissertation\Data/diabetes_data.csv'

def preprocess(path):
    with open(path, 'r') as f:
        df = pd.read_csv(path)
    df['Gender'] = df['Gender'].replace('Male', 1)
    df['Gender'] = df['Gender'].replace('Female', 0)
    df.iloc[:, 2:] = df.iloc[:, 2:].replace({'Yes':1, 'No':0, 'Positive':1, 'Negative':0})
    df.iloc[:, 0] = df.iloc[:, 0] / 90

    df_positive = df[df['class']==1]
    df_negative = df[df['class']==0]

    return df, df_positive, df_negative

def split_data(path):
    _, df_positive, df_negative = preprocess(path)

    x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(df_positive.iloc[:, :16], df_positive.iloc[:, 16], test_size=0.3)
    x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(df_negative.iloc[:, :16], df_negative.iloc[:, 16], test_size=0.3)

    x_train, x_test = pd.concat([x_train_p, x_train_n]), pd.concat([x_test_p, x_test_n])
    y_train, y_test = pd.concat([y_train_p, y_train_n]), pd.concat([y_test_p, y_test_n])

    return x_train, x_test, y_train, y_test

# x_train, x_test, y_train, y_test = split_data(path)