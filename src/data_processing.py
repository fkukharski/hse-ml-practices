import pandas as pd
import pandera as pa
from sklearn.model_selection import train_test_split
from typing import Tuple


def data_reading(path: str) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {"url": pa.Column(str), "label": pa.Column(str, pa.Check.isin(["bad", "good"]))}
    )
    df = pd.read_csv(path)
    validated_df = schema.validate(df)
    return validated_df


def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    schema = pa.DataFrameSchema(
        {"url": pa.Column(str), "label": pa.Column(int, pa.Check.isin([0, 1]))}
    )
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True)
    df["label"] = df["label"].map({"bad": 0, "good": 1})
    validated_df = schema.validate(df)
    validated_df.to_csv("../data/interim/clean_data.csv", index=False)
    return validated_df


def tt_split(
    clean_df: pd.DataFrame, test_size: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_schema = pa.DataFrameSchema({"url": pa.Column(str)})
    y_schema = pa.SeriesSchema(int, pa.Check.isin([0, 1]))
    X = clean_df.drop(["label"], axis=1)
    y = clean_df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=test_size, shuffle=True, stratify=y
    )
    val_X_train = X_schema.validate(X_train)
    val_X_test = X_schema.validate(X_test)
    val_y_train = y_schema.validate(y_train)
    val_y_test = y_schema.validate(y_test)
    return val_X_train, val_X_test, val_y_train, val_y_test
