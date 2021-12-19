# import sys
# sys.path.append("../src")
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from conf_matrix import plot_confusion_matrix
from matplotlib import pyplot as plt
from catboost import CatBoostClassifier
from global_ import DIR_PATH

def rand_forest_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> None:
    skf = StratifiedKFold(random_state=17, n_splits=3, shuffle=True)
    param_grid ={"n_estimators": [i for i in range(5, 20, 3)],
                 "criterion": ["entropy"],
                 "max_depth": [i for i in range(2, 11, 3)]}
    clf = RandomForestClassifier(class_weight="balanced")
    clf_cv = GridSearchCV(clf, param_grid, cv=skf, scoring="recall", n_jobs=-1)
    clf_cv.fit(X_train, y_train)
    font = {"size": 15}
    plt.rc("font", **font)
    cnf_matrix = confusion_matrix(y_test, clf_cv.predict(X_test))
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(cnf_matrix, classes=["0", "1"], title="Confusion matrix")
    plt.savefig(DIR_PATH + "src/reports/random_forest/confusion_matrix.png")
    file = open(DIR_PATH + "src/reports/random_forest/report.txt", "w")
    file.write("roc_auc_score (predict_proba):\n")
    file.write(str(roc_auc_score(y_test, clf_cv.predict_proba(X_test)[:, 0])) + "\n")
    file.write("\n")
    file.write(str(classification_report(y_test, clf_cv.predict(X_test))) + "\n")
    clf_cb = CatBoostClassifier()
    params = {"iterations": [i for i in range(140, 150)],
            "learning_rate": [0.1],
            "depth": [3],
            "loss_function": ["Logloss"]}
    clf_grid = GridSearchCV(clf_cb, params, scoring="f1", n_jobs=-1)
    try:
        clf_grid.fit(X_train, y_train)
    except:
        pass
    file.write(str(f"Best quality - {roc_auc_score(y_test, clf_grid.best_estimator_.predict_proba(X_test)[:,1])}") + "\n")
    file.write(str(f"Parameters: {clf_grid.best_params_}") + "\n")
    file.close()