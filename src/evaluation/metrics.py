# this file contains definitions for classic metrics for multiclass classification

import torch
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def accuracy_score_multiclass(y_true, y_pred):
    """
    Calculates the accuracy of the predictions for multiclass classification where y_pred are logits
    """
    y_true = y_true.cpu().numpy()
    #y_pred are logits
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    
    return accuracy_score(y_true, y_pred)

def precision_score_multiclass(y_true, y_pred):
    """
    Calculates the precision of the predictions for multiclass classification
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    return precision_score(y_true, y_pred, average='macro',zero_division=np.nan)

def recall_score_multiclass(y_true, y_pred):
    """
    Calculates the recall of the predictions for multiclass classification
    """
    y_true = y_true.cpu().numpy()
    
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    return recall_score(y_true, y_pred, average='macro', zero_division=np.nan)

def f1_score_multiclass(y_true, y_pred):
    """
    Calculates the f1 score of the predictions for multiclass classification
    """
    y_true = y_true.cpu().numpy()
    
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average='macro')

def roc_auc_score_multiclass(y_true, y_pred):
    """
    Calculates the roc auc score of the predictions for multiclass classification
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    return roc_auc_score(y_true, y_pred, multi_class='ovo')

def confusion_matrix_multiclass(y_true, y_pred):
    """
    Calculates the confusion matrix of the predictions for multiclass classification
    """
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    y_pred = np.argmax(y_pred, axis=1)
    return confusion_matrix(y_true, y_pred)