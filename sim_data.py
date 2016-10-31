import random
import cPickle
import numpy as np

def sim_data(sampleNum = 5):
    simData = []
    # seqLength = abs(int(random.gauss(150, 20)))
    # seqLength = 5
    minLength = 1000
    maxLength = 5000
    minSample = 5
    maxSample = 100

    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength,maxLength)))
        sampleData = [[0,0,0,0]]
        j = 0
        while j< seqTimeLength:
            delta_t = random.randint(minSample,maxSample)
            delta_l = abs(random.gauss(10,3))
            delta_s = random.gauss(0,2)
            delta_c = random.gauss(0,2)
            j += delta_t
            sampleData.append([j,delta_l,delta_s,delta_c])
        simData.append(sampleData)
    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength, maxLength)))
        sampleData = [[0, 0, 0, 0]]
        j = 0
        while j < seqTimeLength:
            delta_t = random.randint(minSample, maxSample)
            delta_l = abs(random.gauss(0, 3))
            delta_s = random.gauss(1, 2)
            delta_c = random.gauss(5, 2)
            j += delta_t
            sampleData.append([j,delta_l,delta_s,delta_c])
        simData.append(sampleData)
    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength,maxLength)))
        sampleData = [[0,0,0,0]]
        j = 0
        while j< seqTimeLength:
            delta_t = random.randint(minSample,maxSample)
            delta_l = abs(random.gauss(0,3))
            delta_s = random.gauss(0,2)
            delta_c = random.gauss(0,2)
            j += delta_t
            sampleData.append([j,delta_l,delta_s,delta_c])
        simData.append(sampleData)
    for i in range(sampleNum):
        seqTimeLength = abs(int(random.randint(minLength, maxLength)))
        sampleData = [[0, 0, 0, 0]]
        j = 0
        while j < seqTimeLength:
            delta_t = random.randint(minSample, maxSample)
            delta_l = abs(random.gauss(10, 3))
            delta_s = random.gauss(1, 2)
            delta_c = random.gauss(2, 2)
            j += delta_t
            sampleData.append([j,delta_l,delta_s,delta_c])
        simData.append(sampleData)
    return simData


if __name__ == '__main__':
    f = open('sim_data_test','w')
    data = sim_data()
    print np.shape(np.array(data))
    cPickle.dump(data,f)