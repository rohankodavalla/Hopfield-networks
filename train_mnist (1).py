# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import network
from keras.datasets import mnist
from skimage.color import rgb2gray
from skimage.transform import resize
import matplotlib.image as img


# Utils
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(3, 3)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')
            
        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')
            
    plt.tight_layout()
    plt.savefig("result_mnist.png")
    plt.show()

def preprocessing(img):
    print(img.shape)
    w, h = img.shape
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def preprocessing1(img, w=128, h=128):
    # Resize image
    print(img.shape)
    img = resize(img, (w,h), mode='reflect')
    print(img.shape)
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    print(shift.shape, (w*h))
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Load data
    zero = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/0.png','w')).copy()
    one = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/1.png','w')).copy()
    two = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/2.png','w')).copy()
    three = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/3.png','w')).copy()
    four = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/4.png','w')).copy()
    six = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/6.png','w')).copy()
    dash = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/-.png','w')).copy()
    nine = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/9.png','w')).copy()

    zerooo = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/zeroooo.JPEG','w')).copy()
    onee = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/test/one.JPEG','w')).copy()
    twoo = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/test/twoo.JPEG','w')).copy()
    threee = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/test/threee.JPEG','w')).copy()
    fourrr = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/test/four_test.jpg','w')).copy()
    sixx = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/test/Sixx.jpg','w')).copy()
    dashh = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/-.png','w')).copy()
    ninee = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/test/9.jpg','w')).copy()


    x_train = [zero,one,two,three, four, six, dash,nine] 
    x_test = [zerooo,onee,twoo,threee,fourrr,sixx,dashh,ninee]
    
    
    data = []
    
    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing1(d) for d in x_train]
    
    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data)
    
    # Make test datalist
    test = []
    test = [preprocessing1(d) for d in x_test]
    
    predicted = model.predict(test, threshold=100, asyn=True)
    print("Show prediction results...")
    plot(data, test, predicted, figsize=(5, 5))
    print("Show network weights matrix...")
    model.plot_weights()
    
if __name__ == '__main__':
    main()
