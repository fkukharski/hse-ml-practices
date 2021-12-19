import data_processing as processing
import data_featuring as featuring
import tfidf_model as tfidf_modeling
import rand_forest_model as rand_forest_modeling
from global_ import DIR_PATH

data = processing.data_reading(DIR_PATH + "data/raw/data.csv")
clean_data = processing.data_cleaning(data)
processed_data = featuring.manual_feature_engineering(clean_data)

X_train, X_test, y_train, y_test = processing.tt_split(clean_data)
tfidf_modeling.tfidf_model(X_train, X_test, y_train, y_test)

X_train, X_test, y_train, y_test = processing.tt_split(processed_data)
rand_forest_modeling.rand_forest_model(X_train, X_test, y_train, y_test)