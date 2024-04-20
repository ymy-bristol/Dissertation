import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split, cross_validate

# path = 'E:\PersonalFiles\杂项\Dissertation\Data/diabetes_data.csv'

# preprocess data
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


# train test split
def split_data(path):
    _, df_positive, df_negative = preprocess(path)

    x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(df_positive.iloc[:, :16], df_positive.iloc[:, 16], test_size=0.3)
    x_train_n, x_test_n, y_train_n, y_test_n = train_test_split(df_negative.iloc[:, :16], df_negative.iloc[:, 16], test_size=0.3)

    x_train, x_test = pd.concat([x_train_p, x_train_n]), pd.concat([x_test_p, x_test_n])
    y_train, y_test = pd.concat([y_train_p, y_train_n]), pd.concat([y_test_p, y_test_n])

    return x_train, x_test, y_train, y_test


# print the best score
def print_best_score(clf,parameters):
    # 输出best score
    print("Best score: %0.3f" % clf.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


# cross validation for basic models
def cross_val(clf, x, y):
    scoring = ['precision_macro', 'recall_macro', 'f1_macro']
    scores = cross_validate(clf, x, y, scoring=scoring, cv=5, n_jobs=-1)
    print('baisc model results:')
    print('\taverage fit time: ', scores['fit_time'].mean())
    print('\taverage score time: ', scores['score_time'].mean())
    print('\taverage precision: ', scores['test_precision_macro'].mean())
    print('\taverage recall: ', scores['test_recall_macro'].mean())
    print('\taverage f1: ', scores['test_f1_macro'].mean())