import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
from data.global_ import DIR_PATH

def data_reading(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True)
    df["label"] = df["label"].map({"bad": 0, "good": 1})
    df.to_csv(DIR_PATH + 'data/interim/clean_data.csv', index=False)
    return df

def tt_split(clean_df: pd.DataFrame, test_size: float=0.3) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = clean_df.drop(["label"], axis=1)
    y = clean_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=test_size, shuffle=True, stratify=y)
    return X_train, X_test, y_train, y_test
