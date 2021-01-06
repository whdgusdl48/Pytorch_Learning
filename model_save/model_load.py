import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import os

print(tf.__version__)
# ----- DataSet Import ------
(x_train, y_train), (x_test,y_test) = cifar10.load_data()

NUM_CLASSES = 10
print(x_train.shape)

# ----- Feature PreProcessing ----

x_train = x_train[:3000].astype('float32') / 255.0
x_test = x_test[:1000].astype('float32') / 255.0

# ----- One hot encoding -------
y_train = to_categorical(y_train[:3000], NUM_CLASSES)
y_test = to_categorical(y_test[:1000], NUM_CLASSES)

print(x_train[54,12,12,1])

model = Sequential([
    Dense(200, activation = 'relu', input_shape=(32,32,3)),
    Flatten(),
    Dense(150, activation = 'relu'),
    Dense(10, activation = 'softmax')
])
optimizer = Adam(lr = 0.0005)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])

loss, acc = model.evaluate(x_test,y_test)
print(loss,acc)

checkpoint_path = '/home/ubuntu/bjh/Gan/model_save/save'
model.load_weights(checkpoint_path)

loss,acc = model.evaluate(x_test,y_test)
print(loss, acc)

new_model = tf.keras.models.load_model('/home/ubuntu/bjh/Gan/model_save/my_model')

print(new_model.summary())

loss,acc = new_model.evaluate(x_test,y_test)
print(loss,acc)


new_model = tf.keras.models.load_model('test.h5',compile=False)

print(new_model.summary())

loss,acc = new_model.evaluate(x_test,y_test)
print(loss,acc)