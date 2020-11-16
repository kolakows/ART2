import numpy as np
from math import *
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def norm(v1):
    return sqrt(np.sum(np.power(v1, 2)))

def maxIgnoreNan(v):
    maxVal = -inf
    maxInd = -1
    for i in range(len(v)):
        if not v[i] == nan and v[i] > maxVal:
            maxVal = v[i]
            maxInd = i
    return maxVal, maxInd 

def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    D = int(D)
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[int(y_pred[i]), int(y_true[i])] += 1
    ind = linear_sum_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size

def show_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.show()

def show_cms(train_true, train_pred, test_true, test_pred):
    train_true = np.array(train_true)
    train_pred = np.array(train_pred)
    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    cm_test = confusion_matrix(test_true, test_pred)    
    cm_train = confusion_matrix(train_true, train_pred)
    fig, axs = plt.subplots(ncols=2)
    sn.heatmap(pd.DataFrame(cm_train), annot=True, fmt='d', ax = axs[0])
    sn.heatmap(pd.DataFrame(cm_test), annot=True, fmt='d', ax = axs[1])
    axs[0].set_title('Train dataset clustering')
    axs[1].set_title('Test dataset clustering')
    for ax in axs:
        ax.xaxis.set_ticklabels([])
        ax.set_xlabel('Assigned cluster')
        ax.set_ylabel('True cluster')
    plt.show()