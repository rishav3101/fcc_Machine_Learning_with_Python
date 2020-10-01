import tensorflow as tf
import pandas as pd
tf.keras.backend.set_floatx('float64')
CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalWidth', 'Species', ]
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
# Lets define some constants to help us later on
train_path = tf.keras.utils.get_file(
    "iris_training.csv", "http://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "http://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES,
                    header=0)
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES,
                   header=0)
# Here we use keras (a module inside of TensorFlow) to grab our datasets and read them into a pandas dataframe

# print(train.head())

# train.head()
train_y = train.pop('Species')
test_y = test.pop('Species')
print(train.head())

print(train.shape)


def input_fn(features, labels, training=True, batch_size=256):
    # Convert the inputs to a dataset
    dataset = tf.data.Dataset.from_tensor_slices(
        (dict(features), labels))

    # shuffle and repat if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()

        return dataset.batch(batch_size)

# input_fn()


# Feature columns describe how to use the input.
my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

# Build a DNN with 2 hidden layers with 30 and 10 hidden nodes each.
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    # Two hidden layers of 30 and 10 nodes respectively.
    hidden_units=[30, 10],
    # The model must choose between 3 classes.
    n_classes=3
)

t = classifier.train(input_fn=lambda: input_fn(
    train, train_y, training=True), steps=5000)

e = classifier.evaluate(input_fn=lambda: input_fn(
    test, test_y, training=True), steps=5000)

print(e)
print(t)
