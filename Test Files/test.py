import helperFunctions
import digitRecognizer
import numpy



def oneHotInverse(y):
    max = -10
    maxIndex = 0
    for i in range(len(y)):
        if y[i] >= max:
            max = y[i]
            maxIndex = i

    return maxIndex

path = 'mnist_train.csv'

x = digitRecognizer.network(path)
for j in range(1):


    correct = 0
    num = 60000
    for i in range(num):

        img = numpy.array(x.dataMatrix[i])
        z, a = x.guess(img)

        guess = oneHotInverse(z[-1])
        y = x.labels[i]
        if guess == y:
            correct += 1
    print("_________________________________________________________________")
    print('Generation ' + str(j) + ':\n')
    print("Initial accuracy: "+ str(correct * 100/ num) + "%")



    x.train(0.001, 60, num)




    correct = 0
    num = 60000
    for i in range(num):

        img = numpy.array(x.dataMatrix[i])
        z, a = x.guess(img)

        guess = oneHotInverse(z[-1])
        y = x.labels[i]
        if guess == y:
            correct += 1

    print("After Training Accuracy: "+ str(correct * 100/ num) + "%")
    print("_________________________________________________________________")


"""



x = digitRecognizer.network(path)


img = numpy.array(x.dataMatrix[0])
z, a = x.guess(img)

guess = oneHotInverse(a[-1])
y = x.labels[0]

aPrev = numpy.matrix(a[-2]).T

dActivationPrev = 5* numpy.subtract(a[-1],  y)
dBias = numpy.matrix(helperFunctions.sigmoidDerivativeV(z[-1]) * dActivationPrev)
dWeight = numpy.dot(aPrev, dBias)

temp = numpy.matrix(a[-3]).T

dActivation, bias, weight = helperFunctions.backPropogate(numpy.transpose(x.layerFourMatrix), temp, z[-2], z[-1], dActivationPrev)
print(weight)
"""