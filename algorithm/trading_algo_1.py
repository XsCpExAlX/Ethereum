from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import pandas as pd
import tensorflow as tf

# Enable more detailed log output
tf.logging.set_verbosity(tf.logging.INFO)

# Define column names of data sets
COLUMNS = ['id', 'trade_px', 'trade_volume',
           'b1', 'b2', 'b3', 'b4', 'b5',
           'a1', 'a2', 'a3', 'a4', 'a5',
           'bq1', 'bq2', 'bq3', 'bq4', 'bq5',
           'aq1', 'aq2', 'aq3', 'aq4', 'aq5',
           'order_date_time', 'trades_date_time']
FEATURES = ['trade_volume',
           'b1', 'b2', 'b3', 'b4', 'b5',
           'a1', 'a2', 'a3', 'a4', 'a5',
           'bq1', 'bq2', 'bq3', 'bq4', 'bq5',
           'aq1', 'aq2', 'aq3', 'aq4', 'aq5']
LABEL = 'trade_px'

# Read the three CSVs into pandas DataFrame
# Check skiprows param to make sure row 2 is being skipped
train_set = pd.read_csv('C:/Users/donut/PycharmProjects/tensorflowexample/exch_gdax_ethusd_snapshot_train.csv',
                           skipinitialspace=True, skiprows=1, names=COLUMNS)
test_set = pd.read_csv('C:/Users/donut/PycharmProjects/tensorflowexample/exch_gdax_ethusd_snapshot_test.csv',
                           skipinitialspace=True, skiprows=1, names=COLUMNS)
predict_set = pd.read_csv('C:/Users/donut/PycharmProjects/tensorflowexample/exch_gdax_ethusd_snapshot_predict.csv',
                           skipinitialspace=True, skiprows=1, names=COLUMNS)
# print(train_set.dtypes)

# Define feature columns for the input data, which formally specify the set of features used for training
feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

# Instantiate a DNNRegressor for the neural network regression model
regressor = tf.estimator.DNNRegressor(feature_columns=feature_cols,
                                      hidden_units=[10, 10],
                                      model_dir='gdax_ethusd_model')

# Build the input function
def get_input_fn(data_set, num_epochs=None, shuffle=False):
    return tf.estimator.inputs.pandas_input_fn(
        x=pd.DataFrame({k: data_set[k].values for k in FEATURES}),
        y=pd.Series(data_set[LABEL].values),
        num_epochs=num_epochs,
        shuffle=shuffle)

# Train Regressor neural network
regressor.train(input_fn=get_input_fn(train_set), steps=20000)

# Evaluate the trained Regressor neural network
ev = regressor.evaluate(input_fn=get_input_fn(test_set), num_epochs=1, shuffle=False)
loss = ev['loss']
tf.summary.scalar('Loss', loss)
print('Loss: {0:f}'.format(loss))