import tensorflow as tf
import numpy as np
import random
import cPickle
def sim_data(sampleNum = 4):
    simData = []
    # seqLength = abs(int(random.gauss(150, 20)))
    # seqLength = 5
    for i in range(sampleNum):
        seqLength = abs(int(random.randint(5,15)))
        sampleData = []
        for j in range(seqLength):
            item = [abs(random.gauss(10,1)),abs(random.gauss(0,1)),
                    abs(random.gauss(0, 1)),abs(random.gauss(3,1)),
                   abs(random.gauss(0, 1))]
            sampleData.append(item)
        simData.append(sampleData)
    for k in range(sampleNum):
        seqLength = abs(int(random.randint(5, 15)))
        sampleData = []
        for j in range(seqLength):
            item = [abs(random.gauss(1,1)),abs(random.gauss(5,1)),
                    abs(random.gauss(10, 1)),abs(random.gauss(1,1)),
                   abs(random.gauss(5, 1))]
            sampleData.append(item)
        simData.append(sampleData)
    return simData

def loopf(prev, i):
    return prev

# Parameters
learning_rate = 0.001
training_epochs = 200
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
seq_input = tf.placeholder(tf.float32, [max_n_steps,batch_size, frame_dim])
# what timesteps we want to stop at, notice it's different for each batch hence dimension of [batch]
# early_stop = tf.placeholder(tf.int32, [batch_size])

# inputs for rnn needs to be a list, each item/frame being a timestep.
# we need to split our input into each timestep, and reshape it because split keeps dims by default

# useful_input = seq_input[:,0:input_length[0]]
useful_input = seq_input[0:input_length[0]]
loss_inputs = [tf.reshape(useful_input, [-1])]
# encoder_inputs = [tf.reshape(useful_input, [-1, frame_dim])]
encoder_inputs = [item for item in tf.unpack(seq_input)]
# encoder_inputs = [tf.squeeze(item, [1]) for item in tf.split(1, input_length[0],useful_input)]
# encoder_inputs = tf.unpack(useful_input)
# encoder_inputs = [tf.squeeze(t, [1]) for t in tf.split(1, input_length[0], useful_input)]
# if encoder input is "X, Y, Z", then decoder input is "0, X, Y, Z". Therefore, the decoder size
# and target size equal encoder size plus 1. For simplicity, here I droped the last one.
decoder_inputs = ([tf.zeros_like(encoder_inputs[0], name="GO")] + encoder_inputs[:-1])
targets = encoder_inputs
# weights = [tf.ones_like(targets_t, dtype=tf.float32) for targets_t in targets]

# basic LSTM seq2seq model
cell = tf.nn.rnn_cell.LSTMCell(size,state_is_tuple=True,use_peepholes=True)
_, enc_state = tf.nn.rnn(cell, encoder_inputs,sequence_length=input_length[0], dtype=tf.float32)
cell = tf.nn.rnn_cell.OutputProjectionWrapper(cell, frame_dim)
dec_outputs, dec_state = tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell,loop_function=loopf)

# e_stop = np.array([1, 1])

# flatten the prediction and target to compute squared error loss
y_true = [tf.reshape(encoder_input, [-1]) for encoder_input in encoder_inputs]
y_pred = [tf.reshape(dec_output, [-1]) for dec_output in dec_outputs]

# Define loss and optimizer, minimize the squared error
loss = 0
for i in range(len(loss_inputs)):
    loss += tf.reduce_sum(tf.square(tf.sub(y_pred[i], y_true[i])))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    # input_datas = sim_data(4)
    input_datas = cPickle.load(open('behavior_sequences'))
    trajectoryVecs = []
    j = 0
    for input_data in input_datas:
        print 'Sample:'
        print j
        input_len = len(input_data)
        print input_len
        defalt = []
        for i in range(0,frame_dim):
            defalt.append(0)
        while len(input_data)<max_n_steps:
            input_data.append(defalt)
        x = np.array(input_data)
        print np.shape(x[0])
        # print x
        # x = np.arange(n_steps * batch_size * frame_dim)
        # x = x.reshape((batch_size, max_n_steps , frame_dim))
        x = x.reshape((max_n_steps, batch_size, frame_dim))
        embedding = None
        for epoch in range(training_epochs):
            # rand = np.random.rand(n_steps, batch_size, frame_dim).astype('float32')
            # x = np.arange(n_steps * batch_size * frame_dim)
            # x = x.reshape((n_steps, batch_size, frame_dim))
            feed = {seq_input: x,input_length: np.array([input_len])}
            # Fit training using batch data
            _, cost_value, embedding, en_int, de_outs, loss_in = sess.run([optimizer, loss, enc_state,encoder_inputs,dec_outputs,loss_inputs], feed_dict=feed)
             # Display logs per epoch step
            if epoch % display_step == 0:
                # print sess.run(decoder_inputs, feed_dict=feed)
                print "logits"
                a = sess.run(y_pred, feed_dict=feed)
                # print a
                print "labels"
                b = sess.run(y_true, feed_dict=feed)
                # print b

                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(cost_value))
                # print embedding
                # print 'Shape of embedding:'
                # print np.shape(embedding)
                # print 'Shape of en_input:'
                # print np.shape(en_int)
                # print 'Shape of de_outs'
                # print np.shape(de_outs)
                # print 'loss in :'
                # print np.shape(loss_in)
        trajectoryVecs.append(embedding)
        print("Optimization Finished!")
        j=j+1
    fout = file('./sim_data_out_test_normal', 'w')
    cPickle.dump(trajectoryVecs, fout)