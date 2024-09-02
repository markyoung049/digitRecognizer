import math
import numpy as np


def oneHotInverse(y):
    max = -10
    maxIndex = 0
    for i in range(len(y)):
        if y[i] >= max:
            max = y[i]
            maxIndex = i

    return maxIndex

def sigmoid_(x):
  if x > 0:
      
    return 1 / (1 + math.exp(-x))
  else:
      return math.exp(x)/(1+math.exp(x))
  
def relu(x):
	return max(0.0, x)
    
def reluDerivative(x):
    if x > 0:
        return 1
    else:
        return 0

reluDerivativeV = np.vectorize(reluDerivative) 

def sigmoidDerivative(x):
    return sigmoid_(x)* (1-sigmoid_(x))

def sigmoidInverse(x):
    return -np.log((1 / (x + 1e-8)) - 1)

sigmoidDerivativeV = np.vectorize(sigmoidDerivative)    
def backPropogate(w1, z0, z1, z2, y):
    

    dActivationPrev = np.dot(w1, reluDerivativeV(z2) * y)
    bias = reluDerivativeV(z1) * dActivationPrev
    biasM = np.matrix(bias)
    weight = np.dot(z0, biasM)
    return dActivationPrev, bias, weight
    

def gradientDescent(step, vector, filepath):

    layerOne, layerTwo, layerThree, layerFour = grabData(filepath)

    layerOneMatrix = np.array(layerOne[:-1], dtype=float)
    layerTwoMatrix = np.array(layerTwo[:-1], dtype=float)
    layerThreeMatrix = np.array(layerThree[:-1], dtype=float)
    layerFourMatrix = np.array(layerFour[:-1], dtype=float)

    layerOneBias = np.array(layerOne[-1], dtype=float)
    layerTwoBias = np.array(layerTwo[-1], dtype=float)
    layerThreeBias = np.array(layerThree[-1], dtype=float)
    layerFourBias = np.array(layerFour[-1], dtype=float)
    
    layerOneMatrixList = np.subtract(layerOneMatrix,  step * vector[0].T).tolist()
    layerOneBiasList = np.subtract(layerOneBias,  step * vector[1].T).tolist()

    layerTwoMatrixList = np.subtract(layerTwoMatrix,  step * vector[2].T).tolist()
    layerTwoBiasList = np.subtract(layerTwoBias,  step * vector[3].T).tolist()

    layerThreeMatrixList = np.subtract(layerThreeMatrix,  step * vector[4].T).tolist()
    layerThreeBiasList = np.subtract(layerThreeBias,  step * vector[5].T).tolist()

    layerFourMatrixList = np.subtract(layerFourMatrix,  step * vector[6].T).tolist()
    layerFourBiasList = np.subtract(layerFourBias,  step * vector[7].T).tolist()

    network = open(filepath, 'w')
    network.write('Layer 1\n')
    for i in layerOneMatrixList:
        network.write(str(i) + '\n')

    network.write(str(layerOneBiasList)+'\n')


    network.write('Layer 2\n')
    for i in layerTwoMatrixList:
        network.write(str(i) + '\n')

    network.write(str(layerTwoBiasList)+'\n')


    network.write('Layer 3\n')
    for i in layerThreeMatrixList:
        network.write(str(i) + '\n')

    network.write(str(layerThreeBiasList)+'\n')


    network.write('Layer 4\n')
    for i in layerFourMatrixList:
        network.write(str(i) + '\n')

    network.write(str(layerFourBiasList)+'\n')
    network.close()
    return [layerOneMatrix, layerOneBias, layerTwoMatrix, layerTwoBias, layerThreeMatrix, layerThreeBias, layerFourMatrix, layerFourBias]



def oneHot(y):
    x = np.zeros(10)
    x[y] = 1
    return x

def grabData(filePath):

    network = open(filePath, 'r')

    networkRaw = network.readlines()

    # Grab the data for Layer 1
    matrixOne = []
    for i in range(1,22):
        raw = networkRaw[i][1:-2].split(', ')
        row = []
        for val in raw:
            row.append(float(val))
            
        matrixOne.append(row)


    # Grab the data for Layer 2
    matrixTwo = []
    for i in range(23,44):
        raw = networkRaw[i][1:-2].split(', ')
        row = []
        for val in raw:
            row.append(float(val))

        matrixTwo.append(row)


    # Grab the data for Layer 3
    matrixThree = []
    for i in range(45,66):
        raw = networkRaw[i][1:-2].split(', ')
        row = []
        for val in raw:
            row.append(float(val))

        matrixThree.append(row)



    # Grab the data for Layer 4
    matrixFour = []
    for i in range(67,78):
        raw = networkRaw[i][1:-2].split(', ')
        row = []
        for val in raw:
            row.append(float(val))

        matrixFour.append(row)

    


    network.close()

    return matrixOne, matrixTwo, matrixThree, matrixFour

