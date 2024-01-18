import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn

df= pd.read_csv('sudoku.csv')

x= np.array(df.quizzes.map(lambda x: list(map(int ,x))).to_list())
y= np.array(df.solutions.map(lambda x: list(map(int, x))).to_list())

x= x.reshape(-1,9,9,1)
y= y.reshape(-1, 9,9) -1

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(x,y, test_size=0.2, random_state=0)

# cnn model
model= tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same',input_shape=(9,9,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=128,kernel_size=3,activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=256,kernel_size=3,activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=512,kernel_size=3,activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=1024,kernel_size=3,activation='relu',padding='same'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Conv2D(filters=9,kernel_size=3,activation='relu',padding='same'))
model.add(tf.keras.layers.Flatten())
# the below line defines the probablity space (possibility) of the the sudoko space
model.add(tf.keras.layers.Dense(512))
# this below applies because we need to apply the reshape
model.add(tf.keras.layers.Dense(81*9))
model.add(tf.keras.layers.LayerNormalization(axis=-1))
model.add(tf.keras.layers.Reshape((9,9,9)))
model.add(tf.keras.layers.Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
model.summary()

history =model.fit(x_train,y_train, batch_size=64, epochs=15, validation_data=(x_test,y_test))
history

model.evaluate(x_test,y_test)

model.save('sudoku_model.h5')
