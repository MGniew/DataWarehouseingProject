from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.metrics import auc
from sklearn.metrics import roc_curve


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)


def get_classification_report(y_true, y_pred):
    return classification_report(y_true, y_pred)


def get_roc_curve(target, target_score):
    fpr, tpr, _ = roc_curve(target, target_score)
    return fpr, tpr


def get_auc(fpr, tpr):
    return auc(fpr, tpr)
