'''
numpy - Handling array operations and matrix calculations.Flattening and reshaping image vectors.Performing element-wise multiplications for the weight matrix.
matplotlib.pyplot - Plotting images and visualizing the original, noisy, and recalled images.
matplotlib.image - Reading images from files.
Skimage - For preprocessing images, including converting to grayscale, resizing, and binarizing, to prepare them for the Hopfield network.
'''


import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as sp
import matplotlib.image as img
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from skimage.transform import resize
from skimage.filters import threshold_mean


"During training, we create a coefficient matrix that stores the patterns."
"This matrix is updated using the outer product of the image vectors, following the Hebbian learning rule."

def trainer(vector):
    vector = vector.flatten()
    #vector = np.reshape(vector, (len(vector)*len(vector)))
    coefMat = np.zeros([len(vector),len(vector)])
    for i in range(len(vector)):
        for j in range(len(vector)):
            if (i!=(i-j)):
                coefMat[i][i-j] = vector[i]*vector[i-j]
    vector = np.reshape(vector, [int(np.sqrt(len(vector))),int(np.sqrt(len(vector)))])
    return coefMat

"For recalling images, we introduce noise to the input and use the coefficient matrix to reconstruct the original image."
"This is done by iteratively updating the states of neurons based on the weighted sum of inputs from other neurons."

def prediction(curuptedVec,coefMat):
    curuptedVec = curuptedVec.flatten()
    predictVec = np.zeros(len(curuptedVec))
    for i in range(len(curuptedVec)):
        temp = 0
        for j in range(len(curuptedVec)):
             temp += coefMat[i][j] * curuptedVec[j]
        if (temp>0):
            predictVec[i] = 1
        if (temp<0):
            predictVec[i] = -1

    predictVec = np.reshape(predictVec, [int(np.sqrt(len(predictVec))),int(np.sqrt(len(predictVec)))])
    return predictVec
'''
def imageGenerator(imageVector):
    cleanImage = np.zeros([len(imageVector),len(imageVector)])
    for i in range(len(imageVector)):
        for j in range(len(imageVector)):
               cleanImage[i][j] = 1
            else:
                cleanImage[i][j] = -1
    noisyImage = cleanImage + np.random.normal(4, 4, [len(image),len(image)])

    for i in range(len(image)):
        for j in range(len(image)):
            if (noisyImage[i][j] >= 0):
                noisyImage[i][j] = 1
            else:
                noisyImage[i][j] = -1
    return cleanImage,noisyImage


def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data
'''

"Finally, we visualize the results by plotting the original, noisy, and recalled images side by side."
"This helps us see how well the network can recall the original images from corrupted inputs."

def plot(data, test, predicted, figsize=(5, 6)):
    #data = [reshape(d) for d in data]
    #test = [reshape(d) for d in test]
    #predicted = [reshape(d) for d in predicted]

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
    plt.savefig("/content/hopfieldNeuralNetwork/output.png")
    plt.show()


def preprocessing(img, w=64, h=64):
    # Resize image
    img = resize(img, (w,h), mode='reflect')
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    # Reshape
    flatten = np.reshape(img, (w*h))
    return shift

def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted

#Import the image
#image = img.imread('dataset/pgms/bird02.png','w').copy()


'''We first preprocess the images by converting them to grayscale, resizing, and binarizing them. 
This prepares the images for input into the Hopfield network.'''

zero = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/0.png','w')).copy()
one = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/1.png','w')).copy()
two = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/2.png','w')).copy()
three = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/3.png','w')).copy()
four = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/4.png','w')).copy()
six = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/6.png','w')).copy()
dash = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/-.png','w')).copy()
nine = rgb2gray(img.imread('/content/drive/MyDrive/ass 3 nn/9.png','w')).copy()
    
data = [zero, one, two, three, four, six, dash, nine]
print("Start to data preprocessing...")
data = [preprocessing(d) for d in data]

vector_list = []
noisy_list = []
#coefMatrix_list = []
predictedVec_list = []
plt.clf()

plt.figure(figsize=(15,10))
i=0
for image in data :
  #vector,noisyVec = imageGenerator(image)
  noisyVec = get_corrupted_input(image, 0.25)

  coefMatrix = trainer(image)
  predictedVec = prediction(noisyVec,coefMatrix)
  vector_list.append(image)
  noisy_list.append(noisyVec)
  predictedVec_list.append(predictedVec)

#vector,noisyVec = imageGenerator(image)
#coefMatrix = trainer(vector)
#predictedVec = prediction(noisyVec,coefMatrix)

#plot(vector_list, noisy_list, predictedVec_list)


  plt.subplot(8,3,i+1)
  plt.imshow(image)
  if i==0 :

    plt.title('imported picture')
  #plt.subplot(1,4,2)
  #plt.imshow(vector);
  #plt.title('cleaned and croped picture')
  plt.subplot(8,3,i+2)
  plt.imshow(noisyVec);
  if i==0 :
    plt.title('noisy picture')
  plt.subplot(8,3,i+3)
  plt.imshow(predictedVec);
  if i==0 :
    plt.title('recalled picture')
  i=i+3
plt.savefig('hofield.png')
plt.show()
