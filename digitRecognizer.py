# This is the source code for my first machine learning module, where I try to recognize digits
import helperFunctions
import numpy

path = 'mnist_test.csv'
sigmoidv = numpy.vectorize(helperFunctions.sigmoid_)
reluv = numpy.vectorize(helperFunctions.relu)


########################################################################
# Neural Network Class
########################################################################
class network:

    def __init__(self, path):

        # grab the source data
        f = open(path)
        source = f.readlines()[1:]
        f.close()

        data = []
        labels = []

        for line in source:
            # Convert the csv data into lists of integers
            points = line.split(',')

            temp = []

            for i in range(len(points)):
                point = points[i]
                if i == 0:
                    num = int(point.strip())
                else:
                    num = int(point.strip())/255
                temp.append(num)

            points = temp
            
            # Separate image data from labels

            labels.append(points[0])
            points.pop(0)
            data.append(points)

        self.labels = labels
        # initializing matrices
        self.dataMatrix = numpy.array(data, dtype=float)

        layerOne, layerTwo, layerThree, layerFour = helperFunctions.grabData('network.txt')

        self.layerOneMatrix = numpy.array(layerOne[:-1], dtype=float)
        self.layerTwoMatrix = numpy.array(layerTwo[:-1], dtype=float)
        self.layerThreeMatrix = numpy.array(layerThree[:-1], dtype=float)
        self.layerFourMatrix = numpy.array(layerFour[:-1], dtype=float)

        self.layerOneBias = numpy.array(layerOne[-1], dtype=float)
        self.layerTwoBias = numpy.array(layerTwo[-1], dtype=float)
        self.layerThreeBias = numpy.array(layerThree[-1], dtype=float)
        self.layerFourBias = numpy.array(layerFour[-1], dtype=float)


    # method to guess the image based on an image, returns the vector with the guesses
    def guess(self, image):
        results = []
        sums = []
        product=numpy.dot(self.layerOneMatrix, image.transpose())
        sum = numpy.add(product,  self.layerOneBias)
        sums.append(sum)
        results.append(reluv(sum))
        # 2nd Layer
        product=numpy.dot(self.layerTwoMatrix, results[-1])
        sum = numpy.add(product,  self.layerTwoBias)
        sums.append(sum)
        results.append(reluv(sum))
        # 3rd Layer
        product=numpy.dot(self.layerThreeMatrix, results[-1])
        sum = numpy.add(product,  self.layerThreeBias)
        sums.append(sum)
        results.append(reluv(sum))

        # 4th Layer
        product=numpy.dot(self.layerFourMatrix, results[-1])
        sum = numpy.add(product,  self.layerFourBias)
        sums.append(sum)
        results.append(reluv(sum))
        return results, sums



    # Method to train network using data
    def train(self, step, batchSize, dataSize):
        
        # Predict the digit

        for i in range(dataSize//batchSize):
            results = []
            gradients = []
            for j in range(batchSize):
                vector = []
                data = self.dataMatrix[i*batchSize + j]
                # Get the label
                y = helperFunctions.oneHot(self.labels[i*batchSize + j])
                a, z = self.guess(data)
                # Perform the back propogation
                aPrev = numpy.matrix(a[-2]).T

                # First backpropogation step
                dActivationPrev = 5 * numpy.subtract(a[-1],  y)
                dBias = helperFunctions.reluDerivativeV(z[-1]) * dActivationPrev
                dBiasmatrix = numpy.matrix(helperFunctions.reluDerivativeV(z[-1]) * dActivationPrev)
                dWeight = numpy.dot(aPrev, dBiasmatrix)
                
                vector.append(dBias)
                vector.append(dWeight)
                #2nd step
                activationNext = numpy.matrix(a[-3]).T
                dActivationPrev, dBias, dWeight = helperFunctions.backPropogate(numpy.transpose(self.layerFourMatrix), activationNext, z[-2], z[-1], dActivationPrev)
                vector.append(dBias)
                vector.append(dWeight)
                # 3rd step
                activationNext = numpy.matrix(a[-4]).T
                dActivationPrev, dBias, dWeight = helperFunctions.backPropogate(numpy.transpose(self.layerThreeMatrix), activationNext, z[-3], z[-2], dActivationPrev)
                vector.append(dBias)
                vector.append(dWeight)
                #4th step
                activationNext = numpy.matrix(data).T
                dActivationPrev, dBias, dWeight = helperFunctions.backPropogate(numpy.transpose(self.layerTwoMatrix), activationNext, z[-4], z[-3], dActivationPrev)
                vector.append(dBias)
                vector.append(dWeight)

                vector.reverse()
                #add the gradient to the data
                gradients.append(vector)



            # Compute the gradient of the average of the cost
            label = self.labels[i*batchSize + j]
            gradientTotals = gradients[0]
            # Compute the average gradient now
            for k in range(1, len(gradients)):
                gradient = gradients[k]

                for l in range(len(gradient)):
                    gradientTotals[l] = numpy.add(gradient[l], gradientTotals[l])

            gradientAvg = []   
            for m in gradientTotals:
                gradientAvg.append(m * 1/batchSize)
            
            self.updateNetwork(helperFunctions.gradientDescent(step, gradientAvg, 'network.txt'))









    def updateNetwork(self, vector):
        self.layerOneMatrix, self.layerOneBias, self.layerTwoMatrix, self.layerTwoBias, self.layerThreeMatrix, self.layerThreeBias, self.layerFourMatrix, self.layerFourBias = vector[0], vector[1], vector[2], vector[3], vector[4], vector[5], vector[6], vector[7]


        

"""NOTES: For each step, first cvompute dC/daLj. Then, dC/dWjk. Then, dC/dbj"""
