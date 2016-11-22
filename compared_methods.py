import random
import cPickle
import numpy as np
import multiprocessing
import traj_dist.distance as tdist
import os

random.seed(2016)
sampleNum = 10

def trajectoryAlldistance(i,trjs):

    trs_matrix = tdist.cdist(trjs, [trjs[i]],metric="hausdorff")
    cPickle.dump(trs_matrix, open('./distance_compution/hausdorff_distance/hausdorff_distance_' + str(i), 'w'))

    trs_matrix = tdist.cdist(trjs, [trjs[i]],metric="lcss",eps=200)
    cPickle.dump(trs_matrix, open('./distance_compution/LCSS_distance/LCSS_distance_' + str(i), 'w'))
    #
    trs_matrix = tdist.cdist(trjs, [trjs[i]],metric="edr",eps=200)
    cPickle.dump(trs_matrix, open('./distance_compution/EDR_distance/EDR_distance_' + str(i), 'w'))
    #
    trs_matrix = tdist.cdist(trjs, [trjs[i]],metric="dtw")
    cPickle.dump(trs_matrix, open('./distance_compution/DTW_distance/DTW_distance_'+str(i), 'w'))

    print 'complete: '+str(i)


def compute_distance():
    trjs = cPickle.load(open('./simulated_data/sim_trajectories'))
    trs_compare = []
    for tr in trjs:
        trarray = []
        for record in tr:
            trarray.append([record[1],record[2]])
        trs_compare.append(np.array(trarray))
    pool = multiprocessing.Pool(processes=30)
    # print np.shape(distance)
    for i in range(len(trs_compare)):
        print str(i)
        pool.apply_async(trajectoryAlldistance, (i, trs_compare))
    pool.close()
    pool.join()

def combainDistances(inputPath = './distance_compution/DTW_distance/'):
    files = os.listdir(inputPath)
    files_index = []
    for fn in files:
        i = int(fn.split('_')[2])
        files_index.append((fn,i))
    files_index.sort(key=lambda x:x[1])
    distances = []
    for fn in files_index:
        distance = []
        dis = cPickle.load(open(inputPath+fn[0]))
        for i in dis:
            distance.append(i[0])
        distances.append(np.array(distance))
    print np.shape(distances)
    cPickle.dump(distances,open('./distances/'+inputPath.split('/')[2]+'_matrix','w'))

def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')
    # randomly initialize an array of k medoid indices
    M = np.arange(n)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in xrange(tmax):
        # determine clusters, i. e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

def distanceClusterTest(inputFile ='./distances/DTW_distance_matrix'):
    print '---------------------------------'
    print inputFile
    distanceMatrix = cPickle.load(open(inputFile))
    M,C = kMedoids(np.array(distanceMatrix),3)
    cresult = []
    for label in C:
        countStr = 0
        countCir = 0
        countBen = 0
        for point_idx in C[label]:
            if point_idx in range(0,sampleNum): countStr+=1
            if point_idx in range(sampleNum, sampleNum*2): countCir += 1
            if point_idx in range(sampleNum*2, sampleNum*3): countBen += 1
        cresult.append([label,countStr,countCir,countBen])

    all  = 0.

    strList = [[te[0],te[1]] for te in cresult]
    print 'Straight:  '+str(strList)
    m = max([te[1] for te in strList])
    all = all + m
    print float(m) / sampleNum

    cirList = [[te[0],te[2]] for te in cresult]
    print 'Circling:  '+str(cirList)
    m = max([te[1] for te in cirList])
    all = all + m
    print float(m) / sampleNum

    bendList = [[te[0],te[3]] for te in cresult]
    print 'Bending :  '+str(bendList)
    m = max([te[1] for te in bendList])
    all = all + m
    print float(m) / sampleNum
    print 'overall'
    print all/(sampleNum*3)
    print '---------------------------------'

if __name__ == '__main__':
    compute_distance()
    combainDistances(inputPath='./distance_compution/DTW_distance/')
    combainDistances(inputPath='./distance_compution/EDR_distance/')
    combainDistances(inputPath='./distance_compution/LCSS_distance/')
    combainDistances(inputPath='./distance_compution/hausdorff_distance/')
    distanceClusterTest(inputFile='./distances/DTW_distance_matrix')
    distanceClusterTest(inputFile='./distances/EDR_distance_matrix')
    distanceClusterTest(inputFile='./distances/LCSS_distance_matrix')
    distanceClusterTest(inputFile='./distances/hausdorff_distance_matrix')