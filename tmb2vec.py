import random
import cPickle
import numpy as np
import math
import pandas
import tensorflow as tf
from sklearn.cluster import *
from sklearn import preprocessing


random.seed(2016)
sampleNum = 10

def completeTrajectories():
    simTrjss = cPickle.load(open('./simulated_data/sim_trajectories'))
    simTrjComps = []
    for simTrjs in simTrjss:
        trjsCom = []
        for i in range(0,len(simTrjs)):
            rec = []
            if i==0:
                # time, locationC, speedC, rotC
                rec = [0,0,0,0]
            else:
                locC = math.sqrt((simTrjs[i][1]-simTrjs[i-1][1])**2+(simTrjs[i][2]-simTrjs[i-1][2])**2)
                rec.append(simTrjs[i][0])
                rec.append(locC)
                rec.append(locC/(simTrjs[i][0]-simTrjs[i-1][0]))
                rec.append(math.atan((simTrjs[i][2]-simTrjs[i-1][2])/ (simTrjs[i][1]-simTrjs[i-1][1])))
            trjsCom.append(rec)
        simTrjComps.append(trjsCom)
    cPickle.dump(simTrjComps,open('./simulated_data/sim_trajectories_complete','w'))
    return simTrjComps

def computeFeas():
    simTrjCompss = cPickle.load(open('./simulated_data/sim_trajectories_complete'))
    simTrjFeas = []
    for simTrjComps in simTrjCompss:
        trjsComfea = []
        for i in range(0,len(simTrjComps)):
            rec = []
            if i==0:
                # time, locationC, speedC, rotC
                rec = [0,0,0,0]
            else:
                locC = simTrjComps[i][1]
                locCrate = locC/(simTrjComps[i][0]-simTrjComps[i-1][0])
                rec.append(simTrjComps[i][0])
                rec.append(locCrate)
                if locCrate<3:
                    rec.append(0)
                    rec.append(0)
                else:
                    rec.append(simTrjComps[i][2]-simTrjComps[i-1][2])
                    rec.append(simTrjComps[i][3]-simTrjComps[i-1][3])
            trjsComfea.append(rec)
        simTrjFeas.append(trjsComfea)
    cPickle.dump(simTrjFeas, open('./simulated_data/sim_trajectories_feas', 'w'))
    return simTrjFeas

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
    return behavior_sequence

def generate_behavior_sequences():
    f = open('./simulated_data/sim_trajectories_feas')
    sim_data = cPickle.load(f)
    behavior_sequences = []

    for sample in sim_data:
        windows = rolling_window(sample)
        behavior_sequence = behavior_ext(windows)
        print len(behavior_sequence)
        behavior_sequences.append(behavior_sequence)
    fout = open('./simulated_data/sim_behavior_sequences','w')
    cPickle.dump(behavior_sequences,fout)

def generate_normal_behavior_sequence():
    f = open('./simulated_data/sim_behavior_sequences')
    behavior_sequences = cPickle.load(f)

    print np.shape(behavior_sequences)
    behavior_sequences_normal = []
    templist = []
    for item in behavior_sequences:
        for ii in item:
            templist.append(ii)
        print len(item)
    print len(templist)
    min_max_scaler = preprocessing.MinMaxScaler()
    # print np.shape(behavior_sequence)
    templist_normal = min_max_scaler.fit_transform(templist).tolist()
    index = 0
    for item in behavior_sequences:
        behavior_sequence_normal = []
        for ii in item:
            behavior_sequence_normal.append(templist_normal[index])
            index = index + 1
        print len(behavior_sequence_normal)
        behavior_sequences_normal.append(behavior_sequence_normal)
    print index
    print np.shape(behavior_sequences_normal)
    fout = open('./simulated_data/sim_normal_behavior_sequences', 'w')
    cPickle.dump(behavior_sequences_normal, fout)

def trajectory2Vec():
    def loopf(prev, i):
        return prev

    # Parameters
    learning_rate = 0.0001
    training_epochs = 300
    display_step = 100

    # Network Parameters
    # the size of the hidden state for the lstm (notice the lstm uses 2x of this amount so actually lstm will have state of size 2)
    size = 100
    # 2 different sequences total
    batch_size = 1
    # the maximum steps for both sequences is 5
    max_n_steps = 17
    # each element/frame of the sequence has dimension of 3
    frame_dim = 18

    input_length = tf.placeholder(tf.int32)

    initializer = tf.random_uniform_initializer(-1, 1)

    # the sequences, has n steps of maximum size
    # seq_input = tf.placeholder(tf.float32, [batch_size, max_n_steps, frame_dim])
    seq_input = tf.placeholder(tf.float32, [max_n_steps, batch_size, frame_dim])
    # what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]

    # inputs for rnn needs to be a list, each item/frame being a timestep.
    # we need to split our input into each timestep, and reshape it because split keeps dims by default

    useful_input = seq_input[0:input_length[0]]
    loss_inputs = [tf.reshape(useful_input, [-1])]
    encoder_inputs = [item for item in tf.unpack(seq_input)]
    # if encoder input is "X, Y, Z", then decoder input is "0, X, Y, Z". Therefore, the decoder size
    # and target size equal encoder size plus 1. For simplicity, here I droped the last one.
    decoder_inputs = ([tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])
    targets = encoder_inputs

    # basic LSTM seq2seq model
    cell = tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True, use_peepholes=True)
    _, enc_state = tf.nn.rnn(cell, encoder_inputs, sequence_length=input_length[0], dtype=tf.float32)
    cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, frame_dim)
    dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell, loop_function=loopf)


    # flatten the prediction and target to compute squared error loss
    y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
    y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]

    # Define loss and optimizer, minimize the squared error
    loss = 0
    for i in range(len(loss_inputs)):
        loss += tf.reduce_sum(tf.square(tf.sub(y_pred[i], y_true[len(loss_inputs) - i - 1])))
    optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Training cycle
        input_datas = cPickle.load(open('./simulated_data/sim_normal_behavior_sequences'))
        trajectoryVecs = []
        j = 0
        for input_data in input_datas:
            print 'Sample:'
            print j
            input_len = len(input_data)
            print input_len
            defalt = []
            for i in range(0, frame_dim):
                defalt.append(0)
            while len(input_data) < max_n_steps:
                input_data.append(defalt)
            x = np.array(input_data)
            print np.shape(x[0])
            x = x.reshape((max_n_steps, batch_size, frame_dim))
            embedding = None
            for epoch in range(training_epochs):
                feed = {seq_input: x, input_length: np.array([input_len])}
                # Fit training using batch data
                _, cost_value, embedding, en_int, de_outs, loss_in = sess.run(
                    [optimizer, loss, enc_state, encoder_inputs, dec_outputs, loss_inputs], feed_dict=feed)
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print "logits"
                    a = sess.run(y_pred, feed_dict=feed)
                    print "labels"
                    b = sess.run(y_true, feed_dict=feed)

                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(cost_value))
            trajectoryVecs.append(embedding)
            print("Optimization Finished!")
            j = j + 1
        fout = file('./simulated_data/sim_traj_vec_normal_reverse', 'w')
        cPickle.dump(trajectoryVecs, fout)

def vecClusterAnalysis():
    print '---------------------------------'
    print 'Our Method'
    trVecs = []
    trs = cPickle.load(open('./simulated_data/sim_traj_vec_normal_reverse'))
    inte = []
    for tr in trs:
        trVecs.append(tr[0][0])
    km = KMeans(n_clusters=3, random_state=2016)
    clusters = km.fit(trVecs).labels_.tolist()

    all = 0.
    item = set(clusters[:sampleNum])
    l = []
    for i in item:
        l.append([i,clusters[:sampleNum].count(i)])
    print 'Straight:  '+ str(l)
    m = max([te[1] for te in l])
    all = all + m
    print float(m)/sampleNum


    m = 0.
    item = set(clusters[sampleNum:sampleNum*2])
    l = []
    for i in item:
        l.append([i,clusters[sampleNum:sampleNum*2].count(i)])
    print 'Circling:  '+ str(l)
    m = max([te[1] for te in l])
    all = all + m
    print float(m)/sampleNum

    m = 0.
    item = set(clusters[sampleNum*2:sampleNum*3])
    l = []
    for i in item:
        l.append([i,clusters[sampleNum*2:sampleNum*3].count(i)])
    m = max([te[1] for te in l])
    print 'bending:   '+ str(l)
    all = all + m
    print float(m)/sampleNum
    print 'overall'
    print all/(sampleNum*3)
    print '---------------------------------'

if __name__ == '__main__':
    completeTrajectories()
    computeFeas()
    generate_behavior_sequences()
    generate_normal_behavior_sequence()
    trajectory2Vec()
    vecClusterAnalysis()