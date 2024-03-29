from __future__ import absolute_import, division, print_function, unicode_literals
import pathlib

import tensorflow as tf
from tensorflow import keras

import pandas as pd

print(tf.__version__)

# model = keras.models.load_model('my_model.h5')
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

print(predict_raw_dataset)
#
# train_stats = predict_raw_dataset.describe()
# train_stats = train_stats.transpose()
# normed_predict_data=predict_raw_dataset
# print( normed_predict_data)
#
#
# new_model = keras.models.load_model('my_model.h5')
# new_model.summary()
#
# test_predictions = new_model.predict(normed_predict_data)
# print(test_predictions)
# print(new_model.inupt)
# print(new_model.output)

# tf.saved_model.simple_save(
#   tf.keras.backend.get_session(),
#   "./h5_savedmodel/",
#   outputs={"Price": new_model.output}
# )


# with tf.Session(graph=tf.Graph()) as sess:
#     tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING ], 'h5_savedmodel')
#     print(sess)
    # sess.run(normed_predict_data)
