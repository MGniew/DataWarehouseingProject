import itertools

import matplotlib.pyplot as plt
import numpy as np


def curve_plot(fpr, tpr, auc, names, x=[0.0, 1.0], y=[0.0, 1.0]):
    plt.figure()
    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i],
                 label='{} (area = {:3f})'.format(names[i], auc[i]))

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim(x)
    plt.ylim(y)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def heatmap(data):
    plt.figure(dpi=150, figsize=(20, 20))
    data = data.copy()
    data = data.sort_values(by="target")
    plt.imshow(data.T, cmap='hot', interpolation='nearest', aspect=4)
    plt.show()


def scatter_plot(x, y, target):
    plt.scatter(x, y, c=target)
    plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.a
    stollen from sklearn scikit-learn.org
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
