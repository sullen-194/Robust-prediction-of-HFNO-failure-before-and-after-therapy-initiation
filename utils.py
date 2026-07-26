import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from tableone import TableOne

from experiment_config import *

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

def evaluate(y_true, y_preds, y_probs=None, model_label='', verbose=True):
    acc = metrics.accuracy_score(y_true, y_preds)
    auroc = metrics.roc_auc_score(y_true, y_probs[:, 1] if y_probs is not None else y_preds)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_preds).ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    ppv = tp/(tp+fp)
    npv = tn/(tn+fn)

    if verbose:
        print("*** EVALUATION RESULTS ***")
        print(f"{model_label} Accuracy: ", acc)
        print(f"{model_label} AUROC: ", auroc)
        print(f"{model_label} sensitivity: ", sens)
        print(f"{model_label} specificity: ", spec)
        print(f"{model_label} PPV: ", ppv)
        print(f"{model_label} NPV: ", npv)
        print("****************************\n")
    return acc, auroc, sens, spec, ppv, npv


def fit_and_eval_model(X_train, y_train, X_test, y_test, model, model_label='', eval_set=None, verbose=False, plot_conf_matrix=True):
    if verbose:
        model.fit(
            X_train,
            y_train,
            verbose=True
        )
    else:
        model.fit(
            X_train,
            y_train
        )
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)
    acc, auroc, sens, spec, ppv, npv = evaluate(y_test, preds, probs, model_label, verbose=False)

    if plot_conf_matrix:
        cm = metrics.confusion_matrix(y_test, preds)
        score = model.score(X_test, y_test)
        auroc_score = metrics.roc_auc_score(y_test, probs[:, 1])

        # Source: https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
        plt.ylabel('Actual label');
        plt.xlabel('Predicted label');
        all_sample_title = '({0}) - Acc: {1}, AUROC: {2}'.format(model_label, round(score, 4), round(auroc_score, 4))
        plt.title(all_sample_title, size = 15);
        plt.plot()

    return model, preds, acc, auroc, sens, spec, ppv, npv


def print_mc_results(results, n_iter):
    print(f"Monte-Carlo bootstrapping results (#iterations:{n_iter}):")
    for key, scores in results.items():

        mean = np.mean(scores)
        std = np.std(scores)
        result_string = f"{key}: {mean} ({mean - 1.96 * std}--{mean + 1.96 * std})"
        print(result_string)


def mc_bootstrap(X_train, y_train, X_test, y_test, model, model_label='', n_iter=10, eval_set=None, verbose=False):
    results = {
        'acc': [],
        'auroc': [],
        'sens': [],
        'spec': []
    }
    for i in tqdm(range(n_iter)):
        # First, sample from training data with replacement:
        n_samples = len(y_train)
        indices = np.random.randint(0, n_samples, n_samples)
        X_train_sampled = X_train[indices]
        y_train_sampled = y_train[indices]

        # Then, train models with sampled training data to obtain results for Confidence Intervals:
        _, preds, acc, auroc, sens, spec, ppv, npv = fit_and_eval_model(X_train_sampled, y_train_sampled, X_test, y_test, model, model_label, eval_set, verbose, plot_conf_matrix=False)

        results['acc'].append(acc)
        results['auroc'].append(auroc)
        results['sens'].append(sens)
        results['spec'].append(spec)
    
    # Print bootstrapping results:
    print_mc_results(results, n_iter)

    # Finally, train entire model one final time with full dataset:
    model, preds, acc, auroc, sens, spec, ppv, npv = fit_and_eval_model(X_train, y_train, X_test, y_test, model, model_label, eval_set, verbose)
    
    # Print results for full training run
    print("\nFull training run results:")
    print(f"Accuracy: {acc}")
    print(f"AUROC: {auroc}")
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"PPV: {ppv}")
    print(f"NPV: {npv}\n")

    return model, preds, results


def undummify(df_orig, prefixes=[]):
    df = df_orig.copy()
    for prefix in prefixes:
        col_labels = [x for x in df.columns if prefix in x]
        values = df[col_labels].idxmax(axis=1).str.split('_').str[-1].tolist()
        df[prefix] = values

        df = df.drop(columns=col_labels)
    return df


def pretty_print_tableone(df, columns, categorical, groupby, undummify_cols=[], ignore_cols=[]):
    df_data = df.copy()
    ordered_cols = [col for col in list(FORMAT_PARAMS['labels'].keys()) if col not in ignore_cols]

    if set(columns) == set(ordered_cols):
        print("Replacing columns with ordered_cols.")
        columns = ordered_cols
    else:
        print("There was a mismatch between columns and the specified ordered columns. Defaulting to given columns order.")

    if len(undummify_cols) > 0:
        df_data = undummify(df_data, undummify_cols)
    df_data = df_data[columns + [groupby]]

    return TableOne(
        data=df_data,
        columns=columns,
        categorical=categorical,
        nonnormal=NON_NORMAL_VARS,
        groupby=groupby,
        limit=FORMAT_PARAMS['limit'],
        order=FORMAT_PARAMS['order'],
        labels=FORMAT_PARAMS['labels'],
        decimals=FORMAT_PARAMS['decimals'],
        sort=False,
        # htest_name=True,
        pval=True
    )


        
        


