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

# Using Sequence
model = Sequential([
    Dense(200, activation = 'relu', input_shape=(32,32,3)),
    Flatten(),
    Dense(150, activation = 'relu'),
    Dense(10, activation = 'softmax')
])

# Funtional API
input_layer = Input(shape = (32,32,3))

x = Dense(200, activation = 'relu')(input_layer)
x = Flatten()(x)
x = Dense(150, activation = 'relu')(x)
output_layer = Dense(10, activation = 'softmax')(x)

api_model = Model(input_layer,output_layer)

# print(model.summary())
# print(api_model.summary())

# ---- checkpoint ----

# checkpoint_path = '/home/ubuntu/bjh/Gan/model_save/save'
# dir = checkpoint_path
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=dir,save_weights_only=True,verbose=1)


optimizer = Adam(lr = 0.0005)
# model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
# model.fit(x_train,y_train,batch_size=64,epochs=10, shuffle=True,callbacks=[cp_callback])

# model.save('/home/ubuntu/bjh/Gan/model_save/my_model')

checkpoint_path2 = '/home/ubuntu/bjh/Gan/model_save/save2'
dir2 = checkpoint_path2
cp_callback2 = tf.keras.callbacks.ModelCheckpoint(filepath=dir2,save_weights_only=True,verbose=1,period=5)

api_model.compile(
    loss='categorical_crossentropy', 
    optimizer = optimizer, 
    metrics=['accuracy'])

api_model.save_weights(checkpoint_path2.format(epoch=0))
api_model.fit(x_train,y_train,
            batch_size=64,
            epochs=10,
            callbacks=[cp_callback2])

# api_model.save('/home/ubuntu/bjh/Gan/model_save/my_model2')
api_model.save('test.h5')
classes = np.array(['airplain','automobile','bird','cat','deer','dog','frog','horse','ship','truck'])
preds = model.predict(x_test)
preds_single = classes[np.argmax(preds,axis=-1)]
actual_single = classes[np.argmax(y_test, axis=-1)]


