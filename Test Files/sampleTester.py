import digitRecognizer
import random
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image


# Function
def oneHotInverse(y):
    max = -10
    maxIndex = 0
    for i in range(len(y)):
        if y[i] >= max:
            max = y[i]
            maxIndex = i

    return maxIndex



####################################################################################
####################################################################################
####################################################################################
####################################################################################





path = 'mnist_test.csv'

x = digitRecognizer.network(path)

looping = True
correct = 0

for i in range(10000):
    guess, z = x.guess(x.dataMatrix[i])
    guess = oneHotInverse(guess[-1])

    if guess == x.labels[i]:
        correct += 1
print("Accuracy: " + str(correct * 100 / 10000) + '%')


while looping:
    index = random.randint(0, 10000)

    img = np.zeros( (28,28), dtype=float )

    data = x.dataMatrix[index]
    for i in range(28):
        for j in range(28):

            img[i][j] = data[i*28 + j] * 255
    

    first_image = np.array(img, dtype='float')
    
    pixels = first_image.reshape((28, 28))

    plt.imshow(pixels, cmap='gray')
    plt.show()
            
    guess, z = x.guess(data)
    print('_______________________________________________________')
    print("Label is: "+ str(x.labels[index]))
    print("Guess is: "+ str(oneHotInverse(guess[-1])))
    q = input("Press enter to continue, or type q to quit\n")
    print('_______________________________________________________')

    if q == "q":
        looping = False















