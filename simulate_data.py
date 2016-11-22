import random
import cPickle
import numpy as np
import math

random.seed(2016)
sampleNum = 10
speedPreSec = 5
secPreCircle = 3000
a = 10000
b = 500
def sim_data(sampleNum = sampleNum,speedPreSec = speedPreSec,
    secPreCircle = secPreCircle, a = a, b = b):
    ta = math.pi*2
    noise = 50
    minLength = 2500
    maxLength = 5000
    minSample = 20
    maxSample = 50
    simData = []

    a = a
    b = b
    corrInCircleX = []
    corrInCircleY = []
    for i in range(secPreCircle):
        corrInCircleX.append(a*(math.sin(((i*2*math.pi)/secPreCircle)+1.5*math.pi)+1))
        corrInCircleY.append(b*math.cos(((i*2*math.pi)/secPreCircle)+0.5*math.pi))
    # plt.plot(corrInCircleX,corrInCircleY,'*')
    # plt.show()

    # genreate Straight
    # timeInterval, x, y
    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength,maxLength)))
        sampleData = [[0,0,0]]
        j = 0
        previous = [0,0,0]
        while j< seqTimeLength:
            delta_t = random.randint(minSample,maxSample)
            x = previous[1]+ random.gauss(delta_t*speedPreSec,noise)
            y = random.gauss(0,1)
            j += delta_t
            sampleData.append([j,x,y])
            previous = [j,x,y]
        angle = random.random()*ta
        turnSample = []
        for point in sampleData:
            x = point[1]
            y = point[2]
            x1 = math.cos(angle) * x - math.sin(angle) * y
            y1 = math.cos(angle) * y + math.sin(angle) * x
            turnSample.append([point[0],x1,y1])
        simData.append(turnSample)

    # genreate Circling
    # timeInterval, x, y
    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength,maxLength)))
        sampleData = [[0,0,0]]
        j = 0
        while j< seqTimeLength:
            delta_t = random.randint(minSample,maxSample)
            x = random.gauss(corrInCircleX[j%secPreCircle],noise)
            y = random.gauss(corrInCircleY[j%secPreCircle],noise)
            j += delta_t
            sampleData.append([j,x,y])
        angle = random.random()*ta
        turnSample = []
        for point in sampleData:
            x = point[1]
            y = point[2]
            x1 = math.cos(angle) * x - math.sin(angle) * y
            y1 = math.cos(angle) * y + math.sin(angle) * x
            turnSample.append([point[0],x1,y1])
        simData.append(turnSample)
    # genreate Bending
    # timeInterval, x, y
    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength,maxLength)))
        sampleData = [[0,0,0]]
        j = 0
        previous = [0,0,0]
        while j< seqTimeLength:
            delta_t = random.randint(minSample,maxSample)
            x = previous[1]+ random.gauss((delta_t*speedPreSec),noise)
            y = 500 * math.sin(j/(100*math.pi))
            # random.gauss(0,50)
            j += delta_t
            sampleData.append([j,x,y])
            previous = [j,x,y]
        angle = random.random()*ta
        turnSample = []
        for point in sampleData:
            x = point[1]
            y = point[2]
            x1 = math.cos(angle) * x - math.sin(angle) * y
            y1 = math.cos(angle) * y + math.sin(angle) * x
            turnSample.append([point[0],x1,y1])
        simData.append(turnSample)

    cPickle.dump(simData,open('./simulated_data/sim_trajectories','w'))
    return simData

if __name__ == '__main__':
    sim_data()