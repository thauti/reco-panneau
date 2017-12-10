#import tensorflow as tf
import numpy as np
import os
import skimage.data
from skimage.transform import resize
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import load_model
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from keras import backend as K 
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import sys

K.set_image_data_format('channels_first')

def load_image():
    directory = [d for d in os.listdir("Training") if os.path.isdir(os.path.join("Training", d))]
    label = dict()

    imgs = []
    labels = []
    #DEBUG
    #directory = ['00021', '00025', '00019']
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
    y = np.eye(62, dtype="uint8")[labels]
    #print(y[0])
    #print(x)
    return x, y

def load_test():
    directory = [d for d in os.listdir("Testing") if os.path.isdir(os.path.join("Testing", d))]
    label = dict()

    imgs = []
    labels = []
    #DEBUG
    #directory = ['00021', '00025', '00019']
    n = 0
    for a in directory: 
        label[a] = []
        tmp = os.path.join("Testing", a)
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
    x = np.array(imgs)
    y = np.array(labels)
   # print(y)
    #print(x)
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
    model.add(Dense(62, activation='softmax'))
    return model

def lr_schedule(epoch):
    return lr * (0.1 ** int(epoch / 10))

def load_single_image(p):
    return np.rollaxis(resize(skimage.data.imread(p), (64, 64)), -1)

if len(sys.argv) == 3:
    model = load_model(sys.argv[1])
    img = load_single_image(sys.argv[2])
    print(sys.argv[2])
    img =  np.expand_dims(img, axis=0)
    #X, Y  = load_image()

    res = model.predict_classes(img, verbose=1)
    print(res)
else:
    X, Y  = load_image()
    model = setmodel()
    lr = 0.01
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=['accuracy'])

    model.fit(X, Y,
       batch_size=20,
       epochs=10,
       validation_split=0.1,
       callbacks=[LearningRateScheduler(lr_schedule), ModelCheckpoint('model.h5', save_best_only=True)]
    )

    xtest, ytest = load_test()
    print(ytest)
    y_pred = model.predict_classes(xtest)
    acc = np.sum(y_pred == ytest) / np.size(y_pred)
    model.save("result.h5")
    print("Test accuracy = {}".format(acc))
    #print(y_pred)
    #print(ytest)

K.clear_session()