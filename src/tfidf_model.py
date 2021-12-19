# import sys
# sys.path.append("../models")
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import *
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from matplotlib import pyplot as plt
import pandas as pd
from conf_matrix import plot_confusion_matrix


def tfidf_model(
    X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series
) -> None:
    X_train, X_test = X_train["url"], X_test["url"]
    tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=50000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    log_reg = LogisticRegression()
    log_reg.fit(X_train_tfidf, y_train)

    file = open("reports/tfidf/report.txt", "w")
    file.write("roc_auc_score (predict_proba):\n")
    file.write(
        str(roc_auc_score(y_test, log_reg.predict_proba(X_test_tfidf)[:, 1])) + "\n"
    )
    file.write("roc_auc_score (predict):\n")
    file.write(str(roc_auc_score(y_test, log_reg.predict(X_test_tfidf))) + "\n")
    file.write("\n")
    file.write(str(classification_report(y_test, log_reg.predict(X_test_tfidf))) + "\n")
    file.close()
    # print(roc_auc_score(y_test, log_reg.predict_proba(X_test_tfidf)[:, 1]))
    # print(roc_auc_score(y_test, log_reg.predict(X_test_tfidf)))
    # print(classification_report(y_test, log_reg.predict(X_test_tfidf)))

    font = {"size": 15}
    plt.rc("font", **font)
    cnf_matrix = confusion_matrix(y_test, log_reg.predict(X_test_tfidf))
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cnf_matrix, classes=["0", "1"], title="Confusion matrix")
    plt.savefig("reports/tfidf/confusion_matrix.png")
