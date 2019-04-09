""" Module with functions to plot the training results """

import os
from matplotlib import pyplot as plt
import numpy as np
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

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
    
    
    
def plot_discriminant_and_ROC(y_pred, y_test):
    
    if not os.path.exists('Results/'): os.makedirs('Results/')

    discriminant = np.array(y_pred[:,1])
    true_label = np.array(y_test[:,1])


    discriminant0 = discriminant[list(true_labels == 0)]
    discriminant1 = discriminant[list(true_labels == 1)]

    binning = np.linspace(0, 1, 51)

    ### Plot discriminant

    plt.clf()
    plt.figure(num=None, figsize=(6, 6), dpi=600)
    plt.subplot(111)
    pdf0, bins0, patches0 = plt.hist(discriminant0, bins = binning, color = 'r', alpha = 0.3, histtype = 'stepfilled', linewidth = 1, edgecolor='r', label = 'Background')
    pdf1, bins1, patches1 = plt.hist(discriminant1, bins = binning, color = 'b', alpha = 0.3, histtype = 'stepfilled', linewidth = 1, edgecolor='b', label = 'Signal')
    plt.legend(loc = 'upper center')
    plt.ylabel('Entries')
    plt.xlabel('DNN discriminant')
    plt.savefig('Results/Discriminant_distribution.png', dpi = 600)

    ### Plot ROC
    pdf0 /= pdf0.sum()
    pdf1 /= pdf1.sum()

    cum0 = 0
    cum1 = 0


    VPR = []
    FPR = []

    for n in range(0, len(pdf0)):

        cum0 += pdf0[n]
        cum1 += pdf1[n]

        VPR.append(1-cum1)
        FPR.append(1-cum0)

    # Integral under the ROC curve:
    ROC_int = np.trapz(VPR, FPR) # Trapezoidal rule to compute the AUC

    plt.clf()
    plt.figure(num=None, figsize=(6, 6), dpi=600)
    plt.subplot(111)
    plt.plot(FPR, VPR, color = 'r')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.text(0.7, 0.05, 'AUC: %.3f' % abs(ROC_int))
    plt.savefig('Results/ROC.png', dpi =  600)
