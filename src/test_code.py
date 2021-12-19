import os
import numpy as np
from hypothesis.extra.pandas import data_frames , column, range_indexes
from hypothesis import given, settings, strategies as st
from hypothesis import provisional as pr
from hypothesis.strategies._internal.core import characters
from global_ import DIR_PATH
from data_processing import data_reading, data_cleaning, tt_split
from data_featuring import manual_feature_engineering

def test_data_reading():
    assert os.path.isfile(DIR_PATH + "data/raw/data.csv")
    assert len(data_reading(DIR_PATH + "data/raw/data.csv")) > 0

raw_data = data_frames(columns=[
    column(name="url", elements=st.text(alphabet=characters(min_codepoint=65, max_codepoint=122), min_size=5, max_size=10)),
    column(name="label", elements=st.sampled_from(["bad", "good"]))],
    index=range_indexes(min_size=10, max_size=25)
    )

@given(df=raw_data)
@settings(max_examples=250)
def test_data_cleaning(df):
    validated_df = data_cleaning(df)
    assert os.path.isfile(DIR_PATH + "data/interim/clean_data.csv")
    assert len(validated_df) == len(df.drop_duplicates())
    assert sorted(validated_df["label"].unique()) in np.array([0, 1])

clean_data = data_frames(columns=[
    column(name="url", elements=pr.urls()),
    column(name="label", elements=st.just(0))],
    index=range_indexes(min_size=10, max_size=10)
    )

@given(df=clean_data)
@settings(max_examples=250)
def test_tt_split(df):
    df.iloc[:5, 1] = np.array([1, 1, 1, 1, 1])
    X_train, X_test, y_train, y_test = tt_split(df)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == 1
    assert X_test.shape[1] == 1

@given(df=clean_data)
@settings(max_examples=500)
def test_manual_feature_engineering(df):
    returned_df = manual_feature_engineering(df)
    assert returned_df.shape[0] == df.shape[0]
    assert returned_df.shape[1] == 8
    assert returned_df["url_len"].dtype == int
    assert returned_df["num_digits_dom"].dtype == int
    assert returned_df["num_@"].dtype == int
    assert returned_df["num_slash"].dtype == int
    assert os.path.isfile(DIR_PATH + "data/processed/processed_data.csv")