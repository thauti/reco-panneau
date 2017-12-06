import tensorflow as tf
import os
import skimage.data
from skimage.transform import resize

def load_image():
    directory = [d for d in os.listdir("Training") if os.path.isdir(os.path.join("Training", d))]
    label = dict()
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
            label[a].append(resize(skimage.data.imread(c), (64, 64)))
            print(label[a][0].shape)
        n += len(files)
    print(str(n)+" images loaded")

load_image()