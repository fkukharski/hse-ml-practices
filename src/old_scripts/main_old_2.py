import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import tldextract
import ipaddress as ip
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix
from sklearn.metrics import classification_report
import itertools
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from matplotlib import pyplot as plt
import seaborn as sns
from pylab import rcParams

rcParams["figure.figsize"] = 15, 10

# Вероятно нужно импортнуть, чтобы варнинги не мешали, но непонятно
# import warnings
# warnings.filterwarnings("ignore")

# Данные взяты из
# https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs
# репозитория на Github. Автор собирал адреса из открытых списков
# опасных и безопасных сайтов.

# Описание набора данных
# =============================================================================

# Набор данных состоит из двух столбцов:
# url - строка с адресом сайта (не содержит названия протокола)
# label - значение целевой переменной, содержит метки good и bad,
# определяющие безопасен сайт или нет.
# Будем решать задачу бинарной классификации.

df = pd.read_csv("data.csv")
df.info()

# Всего в датасете 420464 строки и отсутствуют пропущенные значения.
# Проверим, есть ли в датасете дубликаты.

df.shape[0] - df.drop_duplicates().shape[0]

# В датасете 9216 дубликатов. Удалим их.

df.drop_duplicates(inplace=True)
df.reset_index(drop=True)

# Заменим bad на 0, good на 1 в столбце целевого признака.

df["label"] = df["label"].map({"bad": 0, "good": 1})

X = df.drop(["label"], axis=1)
y = df["label"]
print(f"Признаков класса 0: {np.sum(y.to_numpy() == 0)}")
print(f"Признаков класса 1: {np.sum(y.to_numpy() == 1)}")
if np.sum(y.to_numpy() == 0) > np.sum(y.to_numpy() == 1):
    print(
        f"Признаков класса 0 в {np.sum(y.to_numpy()==0) /	np.sum(y.to_numpy()==1):.2f} раз больше"
    )
else:
    print(
        f"Признаков класса 1 в {np.sum(y.to_numpy()==1) /	np.sum(y.to_numpy()==0):.2f} раз больше"
    )

# Видна явная несбалансированость классов.
# Позже, можно будет использовать методы для борьбы с ней.
# Пока разделим наш датасет на обучающий и тестовый набор данных.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.3, shuffle=True, stratify=y
)

# Проведем EDA
# =============================================================================

# У нас только один признак - url.
# Попробуем выделить из него дополнительные признаки.

# Предположения:
# 1) Возможно длина адреса влияет на целевую переменную.
#    Предположим, что длинные ссылки чаще явлются вредоносными;
# 2) Количество цифр в адресе тоже может влиять на подозрительность ссылки.
#    Чем больше цифр, тем менее понятно, куда перенаправляет нас ссылка;
# 3) Если вместо адреса используется IP, возможно это вредоносная ссылка;
# 4) Будем проверять содержить ли url запрос;
# 5) Проверка на наличие '@'.
#    Браузер игнорирует все, что находится перед данным символом;
# 6) Наличие в домене букв в верхнем регистре.

urlparse(X_train["url"].iloc[2])

# Так как urllib не выводит отдельно домен, то воспользуемся библиотекой
# tldextract: https://pypi.org/project/tldextract/

tldextract.extract(X_train["url"].iloc[2])

# Также используется библиотека ipaddress, где есть метод ip_address,
# который вернет ошибку, если будет подан не ip-адрес.


def check_ip(domain):
    try:
        if ip.ip_address(domain):
            return 1
    except:
        return 0


df_1 = df.copy()
df_1["url_len"] = df_1["url"].apply(len)
df_1["tfldextract"] = df_1["url"].apply(tldextract.extract)
df_1["num_digits_dom"] = df_1["tfldextract"].apply(
    lambda x: len(re.findall("(\d+)", x.domain))
)
df_1["num_@"] = df_1["url"].apply(lambda x: x.count("@"))
df_1["num_slash"] = df_1["url"].apply(lambda x: x.count("/"))
df_1["query"] = (
    df_1["url"].apply(urlparse).apply(lambda x: 0 if x.query == "" else 1)
)
df_1["caps"] = df_1["tfldextract"].apply(
    lambda x: 1 if len(re.compile("[A-Z]+").findall(x.domain)) > 0 else 0
)
df_1["domain_ip"] = df_1["tfldextract"].apply(lambda x: check_ip(x.domain))

df_1.groupby("num_digits_dom")["url"].nunique()

# Посмотрим визуализацию зависимостей полученных признаков
# =============================================================================

sns.kdeplot(
    df_1[df_1["label"] == 0]["url_len"], color="g", shade=True, label="Bad"
)
sns.kdeplot(
    df_1[df_1["label"] == 1]["url_len"], color="b", shade=True, label="good"
)
plt.title(f"Распределение длины URL")
plt.xlim(0, 500)
plt.legend()
plt.savefig("1.png")
plt.show()

print("Количество плохих адресов с длиной больше 500 символов", end=": ")
print(df_1["url"][(df_1["url_len"] > 500) & (df_1["label"] == 0)].count())

print("Количество хороших адресов с длиной больше 500 символов", end=": ")
print(df_1["url"][(df_1["url_len"] > 500) & (df_1["label"] == 1)].count())

sns.kdeplot(
    df_1[df_1["label"] == 0]["num_digits_dom"],
    color="g",
    shade=True,
    label="Bad",
)
sns.kdeplot(
    df_1[df_1["label"] == 1]["num_digits_dom"],
    color="b",
    shade=True,
    label="good",
)
plt.title("Распределение количества цифр в домене")
plt.xlim(0, 6)
plt.ylim(0, 5)
plt.legend()
plt.savefig("2.png")
plt.show()

tmp1 = sum(df_1[df_1["label"] == 0]["num_digits_dom"] > 1)
print(f"Количество вредоносных сайтов с количеством цифр в домене > 1: {tmp1}")

tmp2 = sum(df_1[df_1["label"] == 1]["num_digits_dom"] > 1)
print(f"Количество безопасных сайтов с количеством цифр в домене > 1: {tmp2}")

sns.barplot(x="label", y="num_@", data=df_1)
plt.title('Зависимость использования "@"')
plt.savefig("3.png")
plt.show()

sns.barplot(x="label", y="num_slash", data=df_1)
plt.title('Зависимость использования "/"')
plt.savefig("4.png")
plt.show()

sns.barplot(x="label", y="domain_ip", data=df_1)
plt.title("Зависимость домена - IP")
plt.savefig("5.png")
plt.show()

sns.barplot(x="label", y="query", data=df_1)
plt.title("Зависимость наличия запроса")
plt.savefig("6.png")
plt.show()

sns.barplot(x="label", y="caps", data=df_1)
plt.title("Зависимость использоание заглавных букв")
plt.savefig("7.png")
plt.show()

# По полученным графикам видно, что сайты с длинным url вероятнее окажутся
# вредоносными, чем сайты с более коротким url,
# но все равно зависимость слишком слабая.
# Но зато можно утверждать, что наличие верхнего регистра, IP вместо домена
# и наличие символа @ указывает на вредоносность сайта, как и предполагалось,
# но в общем объектов с такими признаками небольшое количество.

# Метрики
# =============================================================================

# Будем использовать ROC-AUC и recall для нулевого класса,
# так как важно находить как можно большее количество вредоносных сайтов.
# Также будем выводить confusion matrix.

# Выбор и обучение модели
# =============================================================================

# Так как url являются текстовыми данными, преобразуем их с помощью tfidf.
# Для начала зададим рандомные параметры

X_train, X_test, y_train, y_test = train_test_split(
    df["url"], df["label"], test_size=0.3, random_state=17, stratify=df["label"]
)

tfidf = TfidfVectorizer(ngram_range=(1, 1), max_features=50000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Попробуем обучить логистическую регрессию с параметрами по умолчанию.

print(X_train_tfidf.shape)
log_reg = LogisticRegression()
log_reg.fit(X_train_tfidf, y_train)
roc_auc_score(y_test, log_reg.predict_proba(X_test_tfidf)[:, 1])
roc_auc_score(y_test, log_reg.predict(X_test_tfidf))
print(classification_report(y_test, log_reg.predict(X_test_tfidf)))


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    sns.set_style("dark")
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


font = {"size": 15}
plt.rc("font", **font)

cnf_matrix = confusion_matrix(y_test, log_reg.predict(X_test_tfidf))
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=["0", "1"], title="Confusion matrix")
plt.show()

# Даже с параметрами по умолчанию получился уже достаточно хороший результат.
# Попробуем обучить CatBoost и RandomForest на выделенных ранее признаках.

df_1.drop(["url", "tfldextract"], axis=1, inplace=True)
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    df_1.drop(["label"], axis=1),
    df_1["label"],
    test_size=0.3,
    random_state=42,
    stratify=df_1["label"],
)

skf = StratifiedKFold(random_state=17, n_splits=3, shuffle=True)
param_grid = {
    "n_estimators": [i for i in range(5, 20, 3)],
    "criterion": ["entropy"],
    "max_depth": [i for i in range(2, 11, 3)],
}

clf = RandomForestClassifier(class_weight="balanced")
clf_cv = GridSearchCV(clf, param_grid, cv=skf, scoring="recall", n_jobs=-1)
clf_cv.fit(X_train_f, y_train_f)

roc_auc_score(y_test_f, clf_cv.predict_proba(X_test_f)[:, 0])

cnf_matrix = confusion_matrix(y_test_f, clf_cv.predict(X_test_f))
plt.figure(figsize=(10, 8))
plot_confusion_matrix(cnf_matrix, classes=["0", "1"], title="Confusion matrix")
plt.show()

print(classification_report(y_test_f, clf_cv.predict(X_test_f)))

clf_cb = CatBoostClassifier()
params = {
    "iterations": [i for i in range(100, 150)],
    "learning_rate": [0.1],
    "depth": [3],
    "loss_function": ["Logloss"],
}

clf_grid = GridSearchCV(clf_cb, params, scoring="f1", n_jobs=-1)
clf_grid.fit(X_train_f, y_train_f)

print(
    f"Лучшее качество - {roc_auc_score(y_test_f, clf_grid.best_estimator_ .predict_proba(X_test_f)[:,1])}"
)
print(f"Параметры: {clf_grid.best_params_}")

# Видно, что на выделенных признаках у нас не получается такой точности,
# как с использованием tfidf. Попробуем настроить параметры tfidf,
# учитывать последовательность в url и использовать GridSearch
# для логистической регрессии

url_df = pd.DataFrame()
url_df["extract"] = df["url"].apply(tldextract.extract)
url_df["urlparse"] = df["url"].apply(urlparse)

# достаем 4 текстовых признака

url_df["hostname"] = url_df["urlparse"].apply(lambda x: x.path.split("/")[0])
url_df["suffix"] = url_df["extract"].apply(lambda x: x.suffix.replace(".", " "))
url_df["path"] = url_df["urlparse"].apply(
    lambda x: "".join([s + " " for s in x.path.split("/")[1:]]).strip()
)
url_df["domain"] = url_df["extract"].apply(lambda x: x.domain)
url_df.drop(["extract", "urlparse"], axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(
    url_df,
    df_1["label"],
    test_size=0.3,
    random_state=42,
    stratify=df_1["label"],
)


def to_tfidf(X_train1, X_test1, n, max_feat):
    tfidf = TfidfVectorizer(ngram_range=(1, n), max_features=max_feat)
    train_1 = tfidf.fit_transform(X_train1["hostname"])
    test_1 = tfidf.transform(X_test1["hostname"])
    train_2 = tfidf.fit_transform(X_train1["suffix"])
    test_2 = tfidf.transform(X_test1["suffix"])
    train_3 = tfidf.fit_transform(X_train1["path"])
    test_3 = tfidf.transform(X_test1["path"])
    train_4 = tfidf.fit_transform(X_train1["domain"])
    test_4 = tfidf.transform(X_test1["domain"])
    X_train_sparse_cv = csr_matrix(np.hstack([train_1, train_2, train_3, train_4]))
    X_test_sparse_cv = csr_matrix(np.hstack([test_1, test_2, test_3, test_4]))
    return X_train_sparse_cv, X_test_sparse_cv


X_train_tfidf, X_test_tfidf = to_tfidf(X_train, X_test, 1, 50000)

param_grid = {"C": [0.01, 0.05, 0.1, 0.5, 1, 5, 10]}
log_reg = LogisticRegression(random_state=42)
log_reg_cv = GridSearchCV(log_reg, param_grid, scoring="roc_auc")
log_reg_cv.fit(X_train_tfidf, y_train)


def metrics(X_test_m, y_test_m, model):

    print(roc_auc_score(y_test_m, model.predict_proba(X_test_m)[:, 1]))

    cnf_matrix = confusion_matrix(y_test_m, model.predict(X_test_m))
    plt.figure(figsize=(10, 8))
    plot_confusion_matrix(
        cnf_matrix, classes=["0", "1"], title="Confusion matrix"
    )
    plt.show()
    print(classification_report(y_test_m, model.predict(X_test_m)))


print(metrics(X_test_tfidf, y_test, log_reg_cv))
print(log_reg_cv.best_params_)

score = []
for i in range(1, 3):
    for k in range(50000, 100001, 10000):
        X_train_tfidf, X_test_tfidf = to_tfidf(X_train, X_test, i, k)
        log_reg = LogisticRegression(C=10, random_state=42)
        log_reg.fit(X_train_tfidf, y_train)
        score.append(
            recall_score(y_test, log_reg.predict(X_test_tfidf), pos_label=0)
        )

print(score)
