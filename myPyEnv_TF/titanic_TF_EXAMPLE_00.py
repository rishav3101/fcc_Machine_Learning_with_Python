import tensorflow as tf
import pandas as pd
import pprint

dftrain = pd.read_csv("./data/train.csv")
dfeval = pd.read_csv("./data/eval.csv")

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

CATEGORICAL_COLUMNS = ['sex', 'sibsp',
                       'parch', 'pclass', 'parch', 'embarked', 'boat']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    # gets alist of all uniquue valueus from given feature column
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(
        tf.feature_column.sequence_categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(
        feature_name, dtype=tf.float32))
print(feature_columns)
print(dftrain["embarked"].unique())
