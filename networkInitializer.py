import random
import math
import numpy as np
run = False

if run:
    # Neural Network Structure and starting params
    network = open('network.txt', 'w+')


    # layer one initialization
    layerOne = []
    biasOne = []
    for i in range(0, 20):
        biasOne.append(0)


    for i in range(20):
        row = []
        for j in range(784):
            row.append(random.normalvariate(0, math.sqrt(2/784)))
        
        layerOne.append(row)


    # layer two initialization
    layerTwo = []
    biasTwo = []

    for i in range(0, 20):
        biasTwo.append(0)

    for i in range(20):
        row = []
        for j in range(20):
            row.append(random.normalvariate(0, math.sqrt(2/20)))
        
        layerTwo.append(row)



    # layer three initialization
    layerThree = []
    biasThree = []

    for i in range(0, 20):
        biasThree.append(0)

    for i in range(20):
        row = []
        for j in range(20):
            row.append(random.normalvariate(0, math.sqrt(2/20)))
        
        layerThree.append(row)



    # layer four initialization
    layerFour = []
    biasFour = []

    for i in range(0, 10):
        biasFour.append(0)

    for i in range(10):
        row = []
        for j in range(20):
            row.append(random.normalvariate(0, math.sqrt(2/20)))
        
        layerFour.append(row)

    network.write('Layer 1\n')
    for i in layerOne:
        network.write(str(i) + '\n')

    network.write(str(biasOne)+'\n')


    network.write('Layer 2\n')
    for i in layerTwo:
        network.write(str(i) + '\n')

    network.write(str(biasTwo)+'\n')


    network.write('Layer 3\n')
    for i in layerThree:
        network.write(str(i) + '\n')

    network.write(str(biasThree)+'\n')


    network.write('Layer 4\n')
    for i in layerFour:
        network.write(str(i) + '\n')

    network.write(str(biasFour)+'\n')
    network.close()
    
    print('Done')

else:
    print("Program disabled.")