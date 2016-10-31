import random
import cPickle
import numpy as np
import pandas
from sklearn import preprocessing


def rolling_window(sample, windowsize = 600, offset = 300):
    timeLength = sample[len(sample)-1][0]
    windowLength = int (timeLength/offset)+1
    windows = []
    for i in range(0,windowLength):
        windows.append([])

    for record in sample:
        time = record[0]
        for i in range(0,windowLength):
            if (time>(i*offset)) & (time<(i*offset+windowsize)):
                windows[i].append(record)
    return windows
    # pass

def behavior_ext(windows):
    behavior_sequence = []
    for window in windows:
        behaviorFeature = []
        records = np.array(window)
        if len(records) != 0:
            # print np.shape(records)
            pd = pandas.DataFrame(records)
            pdd =  pd.describe()
            # print pdd[1][0]
            # for ii in range(1,4):
            #     for jj in range(1,8):
            #         behaviorFeature.append(pdd[ii][jj])
            # behaviorFeature.append(pdd[0][1])
            behaviorFeature.append(pdd[1][1])
            behaviorFeature.append(pdd[2][1])
            behaviorFeature.append(pdd[3][1])
            # behaviorFeature.append(pdd[0][2])
            # behaviorFeature.append(pdd[1][2])
            # behaviorFeature.append(pdd[2][2])
            # behaviorFeature.append(pdd[3][2])
            # behaviorFeature.append(pdd[0][3])
            behaviorFeature.append(pdd[1][3])
            behaviorFeature.append(pdd[2][3])
            behaviorFeature.append(pdd[3][3])
            # behaviorFeature.append(pdd[0][4])
            behaviorFeature.append(pdd[1][4])
            behaviorFeature.append(pdd[2][4])
            behaviorFeature.append(pdd[3][4])
            # behaviorFeature.append(pdd[0][5])
            behaviorFeature.append(pdd[1][5])
            behaviorFeature.append(pdd[2][5])
            behaviorFeature.append(pdd[3][5])
            # behaviorFeature.append(pdd[0][6])
            behaviorFeature.append(pdd[1][6])
            behaviorFeature.append(pdd[2][6])
            behaviorFeature.append(pdd[3][6])
            # behaviorFeature.append(pdd[0][7])
            behaviorFeature.append(pdd[1][7])
            behaviorFeature.append(pdd[2][7])
            behaviorFeature.append(pdd[3][7])

            behavior_sequence.append(behaviorFeature)
    min_max_scaler = preprocessing.MinMaxScaler()
    print np.shape(behavior_sequence)
    behavior_sequence_normal = min_max_scaler.fit_transform(behavior_sequence).tolist()

    return behavior_sequence, behavior_sequence_normal
    # pass

if __name__ == '__main__':
    f = open('sim_data_test')
    sim_data = cPickle.load(f)
    behavior_sequences = []
    behavior_sequences_normal = []
    for sample in sim_data:
        windows = rolling_window(sample)
        behavior_sequence, behavior_sequence_normal = behavior_ext(windows)
        print len(behavior_sequence)
        behavior_sequences.append(behavior_sequence)
        behavior_sequences_normal.append(behavior_sequence_normal)
    fout = open('behavior_sequences','w')
    cPickle.dump(behavior_sequences,fout)
    fout = open('behavior_sequences_normal','w')
    cPickle.dump(behavior_sequences_normal,fout)
