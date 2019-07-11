from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

dataset_path ='whPrice.data'
test_dataset_path='whValidPrice.data'
column_names = ['area','category','price']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
test_raw_dataset=pd.read_csv(test_dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.tail()
dataset.isna().sum()
dataset = dataset.dropna()
area = dataset.pop('area')
dataset['江岸区'] = (area == 420102)*1.0
dataset['江汉区'] = (area == 420103)*1.0
dataset['硚口区'] = (area == 420104)*1.0
dataset['汉阳区'] = (area == 420105)*1.0
dataset['武昌区'] = (area == 420106)*1.0
dataset['洪山区'] = (area == 420111)*1.0
dataset['东西湖区'] = (area == 420112)*1.0
dataset['汉南区'] = (area == 420113)*1.0
dataset['蔡甸区'] = (area == 420114)*1.0
dataset['江夏区'] = (area == 420115)*1.0
dataset['黄陂区'] = (area == 420116)*1.0
dataset['新洲区'] = (area == 420117)*1.0
dataset['东湖高新区'] = (area == 420118)*1.0
dataset['经济开发区'] = (area == 420119)*1.0
dataset.tail()
train_dataset = dataset

test_dataset=test_raw_dataset.copy()
test_dataset.tail()
test_dataset.isna().sum()
test_dataset = test_dataset.dropna()
area = test_dataset.pop('area')
test_dataset['江岸区'] = (area == 420102)*1.0
test_dataset['江汉区'] = (area == 420103)*1.0
test_dataset['硚口区'] = (area == 420104)*1.0
test_dataset['汉阳区'] = (area == 420105)*1.0
test_dataset['武昌区'] = (area == 420106)*1.0
test_dataset['洪山区'] = (area == 420111)*1.0
test_dataset['东西湖区'] = (area == 420112)*1.0
test_dataset['汉南区'] = (area == 420113)*1.0
test_dataset['蔡甸区'] = (area == 420114)*1.0
test_dataset['江夏区'] = (area == 420115)*1.0
test_dataset['黄陂区'] = (area == 420116)*1.0
test_dataset['新洲区'] = (area == 420117)*1.0
test_dataset['东湖高新区'] = (area == 420118)*1.0
test_dataset['经济开发区'] = (area == 420119)*1.0
test_dataset.tail()

train_stats = train_dataset.describe()
train_stats.pop("price")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('price')
test_labels = test_dataset.pop('price')

normed_train_data = train_dataset
normed_test_data = test_dataset
def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()
EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
model = build_model()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0)
# print( model.evaluate(normed_test_data, test_labels, verbose=0))
# loss,acc, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# print("Testing set Mean Abs Error: {:5.2f} price".format(mae))

predict_raw_dataset=pd.read_csv('predict.data', names=['area','category'],
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)
area = predict_raw_dataset.pop('area')
predict_raw_dataset['江岸区'] = (area == 420102)*1.0
predict_raw_dataset['江汉区'] = (area == 420103)*1.0
predict_raw_dataset['硚口区'] = (area == 420104)*1.0
predict_raw_dataset['汉阳区'] = (area == 420105)*1.0
predict_raw_dataset['武昌区'] = (area == 420106)*1.0
predict_raw_dataset['洪山区'] = (area == 420111)*1.0
predict_raw_dataset['东西湖区'] = (area == 420112)*1.0
predict_raw_dataset['汉南区'] = (area == 420113)*1.0
predict_raw_dataset['蔡甸区'] = (area == 420114)*1.0
predict_raw_dataset['江夏区'] = (area == 420115)*1.0
predict_raw_dataset['黄陂区'] = (area == 420116)*1.0
predict_raw_dataset['新洲区'] = (area == 420117)*1.0
predict_raw_dataset['东湖高新区'] = (area == 420118)*1.0
predict_raw_dataset['经济开发区'] = (area == 420119)*1.0
predict_raw_dataset.tail()

normed_predict_data=predict_raw_dataset
print(normed_predict_data)
test_predictions = model.predict(normed_predict_data).flatten()
print(test_predictions)
model.save('my_model.h5')

tf.saved_model.simple_save(
  tf.keras.backend.get_session(),
  "./h5_savedmodel/",
  inputs={"BuildingInfo": model.input},
  outputs={"Price": model.output}
)
