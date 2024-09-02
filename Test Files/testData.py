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

path = 'mnist_test.csv'


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

data = numpy.array(data)

x = digitRecognizer.network(path)
correct = 0
num = 10000
for i in range(num):

    img = numpy.array(data[i])
    z, a = x.guess(img)

    guess = oneHotInverse(z[-1])
    y = labels[i]
    if guess == y:
        correct += 1

print("Accuracy: "+ str(correct * 100/ num) + "%")
print("_________________________________________________________________")