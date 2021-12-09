DIR_PATH = '/home/fkukharski/git_hws/hse-ml-practices/'
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
import itertools
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    sns.set_style('dark')
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # if normalize:
    #     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')
    # print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def tfidf_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    X_train, X_test = X_train['url'], X_test['url']
    tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=50000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_tfidf, y_train)

    file = open(DIR_PATH + 'src/reports/tfidf/report.txt', 'w')
    file.write('roc_auc_score (predict_proba):\n')
    file.write(str(roc_auc_score(y_test, log_reg.predict_proba(X_test_tfidf)[:, 1])) + '\n')
    file.write('roc_auc_score (predict):\n')
    file.write(str(roc_auc_score(y_test, log_reg.predict(X_test_tfidf))) + '\n')
    file.write('\n')
    file.write(str(classification_report(y_test, log_reg.predict(X_test_tfidf))) + '\n')
    file.close()
    # print(roc_auc_score(y_test, log_reg.predict_proba(X_test_tfidf)[:, 1]))
    # print(roc_auc_score(y_test, log_reg.predict(X_test_tfidf)))
    # print(classification_report(y_test, log_reg.predict(X_test_tfidf)))
    
    font = {'size' : 15}
    plt.rc('font', **font)
    cnf_matrix = confusion_matrix(y_test, log_reg.predict(X_test_tfidf))
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cnf_matrix, classes=['0', '1'], title='Confusion matrix')
    plt.savefig(DIR_PATH + 'src/reports/tfidf/confusion_matrix.png')