from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import image_loader,image_processing,split_data

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(1)

# ---- image load and processing -------

img_path = os.path.join('/home/ubuntu/bjh/Gan/apple2orange')
img_list = os.listdir(img_path)

IMAGE_SIZE = 128
train_data, test_data = image_loader(img_path,(IMAGE_SIZE,IMAGE_SIZE))

x_train, y_train = train_data[0], train_data[1]
x_test, y_test = test_data[0], test_data[1]

x_train = image_processing(x_train)
x_test = image_processing(x_test)

# ---------------------------------------

# ------- pretraining model -------------

IMAGE_SHAPE = (IMAGE_SIZE,IMAGE_SIZE,3)

vgg = tf.keras.applications.VGG16(input_shape=IMAGE_SHAPE,
                                  include_top=False,
                                  weights= 'imagenet')

vgg.trainable = False

pretrain_model = tf.keras.Sequential([
    vgg,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

print(pretrain_model.summary())

learning_rate = 0.0001
pretrain_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate),
                       loss = 'binary_crossentropy',
                       metrics= ['accuracy'])

checkpoint_path = 'pretrained/pretrained'
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1,
                                                  period=5)

history = pretrain_model.fit(x_train,y_train,
                             epochs=20, 
                             validation_data = (x_test,y_test),
                             callbacks=[cp_callback])

# ---------------------------------------------------

predict = np.argmax(pretrain_model.predict(x_test),axis=1)

fig, axs = plt.subplots(3, 4, figsize=(25,12.5))
cnt = 0
for i in range(3):
    for j in range(4):
        axs[i,j].imshow(x_test[i+j],cmap='gray_r')
        axs[i, j].set_title(predict[i+j])
        axs[i,j].axis('off')
        cnt += 1        
fig.savefig('./' + "%s.png" % ('predict'))

# -----save fig  --------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.savefig('./' + "%s.png" % ('accuarcy'))
