import random

import cPickle
import numpy as np
import numpy as np
import tensorflow as tf

def sim_data(sampleNum = 2):
    simData = []
    for i in range(sampleNum):
        seqLength = abs(int(random.gauss(150, 30)))
        sampleData = []
        for j in range(seqLength):
            item = [abs(random.gauss(10,1)),abs(random.gauss(0,1)),
                    abs(random.gauss(0, 1)),abs(random.gauss(3,1)),
                   abs(random.gauss(0, 1))]
            sampleData.append(item)
        simData.append(sampleData)
    for k in range(sampleNum):
        seqLength = abs(int(random.gauss(150, 20)))
        sampleData = []
        for j in range(seqLength):
            item = [abs(random.gauss(1,1)),abs(random.gauss(5,1)),
                    abs(random.gauss(10, 1)),abs(random.gauss(1,1)),
                   abs(random.gauss(5, 1))]
            sampleData.append(item)
        simData.append(sampleData)
    return simData

def trajectroy_embedding(rowdata,batch_num = 1,hidden_num = 512,elem_num = 5,iteration = 2):
    rowdata = np.array(rowdata)
    step_num = len(rowdata)
    print step_num
    p_input = tf.placeholder(tf.float32, [batch_num, step_num, elem_num])
    p_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, step_num, p_input)]
    cell = tf.nn.rnn_cell.LSTMCell(hidden_num, use_peepholes=True)
    outputs, states = tf.model.rnn.embedding_rnn_seq2seq(p_inputs,p_inputs,cell,dtype=tf.float32)
    # LSTMAutoencoder(hidden_num, p_inputs, cell=cell, decode_without_input=False)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        print index
        for i in range(iteration):
            """Random sequences.
              Every sequence has size batch_num * step_num * elem_num
              Each step number increases 1 by 1.
              An initial number of each sequence is in the range from 0 to 19.
              (ex. [8. 9. 10. 11. 12. 13. 14. 15])
            """
            random_sequences = rowdata.reshape([1, step_num, elem_num])
            # print np.transpose(random_sequences)
            # print random_sequences
            print random_sequences.shape
            loss_val, _, enout = sess.run([ae.loss, ae.train, ae.enout], {p_input: random_sequences})
            print "iter %d:" % (i + 1), loss_val
        return enout