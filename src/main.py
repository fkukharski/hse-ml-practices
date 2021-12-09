DIR_PATH = '/home/fkukharski/git_hws/hse-ml-practices/'

import data.data_processing as processing
import models.tfidf_model as tfidf_modeling

data = processing.data_reading(DIR_PATH + 'data/raw/data.csv')
clean_data = processing.data_cleaning(data)
X_train, X_test, y_train, y_test = processing.tt_split(clean_data)
tfidf_modeling.tfidf_model(X_train, X_test, y_train, y_test)
