#import tensorflow as tf
import numpy as np
import os
import skimage.data
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import SGD
from keras import backend as K 
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
K.set_image_data_format('channels_first')

def load_image():
    directory = [d for d in os.listdir("Training") if os.path.isdir(os.path.join("Training", d))]
    label = dict()

    imgs = []
    labels = []
    #DEBUG
    directory = ['00021', '00025']
    n = 0
    for a in directory: 
        label[a] = []
        tmp = os.path.join("Training", a)
        files = [os.path.join(tmp, f) 
                      for f in os.listdir(tmp) 
                        if f.endswith(".ppm")]
       
        for c in files:
            #label[a].append(resize(skimage.data.imread(c), (64, 64)))
            imgs.append(np.rollaxis(resize(skimage.data.imread(c), (64, 64)), -1))
            labels.append(int(a))
            #print(labels)
            #print(label[a][0].shape)

        n += len(files)
        print(str(n)+" images loaded")
    
    print(len(imgs))
    x = np.array(imgs, dtype="float32")
    y = np.eye(26, dtype="uint8")[labels]
    print(y)
    print(x)
    return x, y

def setmodel():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(3, 64, 64),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation='softmax'))
    return model

X, Y  = load_image()
model = setmodel()
lr = 0.01
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

model.fit(X, Y,
   batch_size=32,
    epochs=30,
    validation_split=0.2,
    callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5', save_best_only=True)]
)
