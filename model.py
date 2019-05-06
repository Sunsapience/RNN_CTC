import tensorflow as tf
import numpy as np 
from tensorflow.contrib import cudnn_rnn, rnn

from configs import cfg
from data_utils import batch_train_flow,batch_test_flow,batch_valid_flow,decode_sparse_tensor

import logging,pickle,time
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class RNNCTC():
    def __init__(self,is_training):
        self.inputs = tf.placeholder(tf.float32, [None, 280, 28])
        self.labels = tf.sparse_placeholder(tf.int32, name='targets')
        self.seq_len = tf.placeholder(tf.int32, [None])
        self.time_inputs = tf.transpose(self.inputs,[1,0,2]) # time_major

        with tf.variable_scope("outputs"):
            rnn_outputs = self.birnn(True)
            logits = tf.reshape(rnn_outputs, (-1, 2*cfg.hidden_size))
            logits = tf.layers.dense(logits,cfg.class_size)
            self.logits = tf.reshape(logits,[cfg.max_time,-1,cfg.class_size])

        with tf.name_scope('losses'):
            self.cost = tf.nn.ctc_loss(labels=self.labels, inputs=self.logits, sequence_length=self.seq_len)
            self.loss = tf.reduce_mean(self.cost)
        
        with tf.name_scope('decoder'):
            self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(self.logits, 
                        self.seq_len, beam_width=20, merge_repeated=False)
            self.err = tf.reduce_mean(tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.labels))
        
        if not is_training:
            return
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate=cfg.learning_rate).minimize(self.loss) 

    def birnn(self,use_cudnn=False):
        if use_cudnn:
            DIRECTION = "bidirectional"
            cell = cudnn_rnn.CudnnLSTM(1,cfg.hidden_size,direction=DIRECTION)
            outputs,_ = cell(self.time_inputs)
        else:
            cell_1 = rnn.BasicLSTMCell(cfg.hidden_size)
            cell_2 = rnn.BasicLSTMCell(cfg.hidden_size)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_1, cell_2, 
                    self.time_inputs, dtype=tf.float32,time_major=True)
            outputs = tf.concat(outputs, 2)
        return outputs
####################################
def run_epoch(sess,Model,data_queue, train_op,train_or):
    i = 0
    b_err = []
    b_loss = []
    for x,y,sparse_y in data_queue:
        feed = {
            Model.inputs: x,
            Model.labels: sparse_y,
            Model.seq_len: [cfg.max_time]*len(x)
        }
        sess.run(train_op, feed_dict=feed)

        if i%10 == 0:
            loss, err, decode = sess.run([Model.loss, Model.err, Model.decoded], 
                    feed_dict=feed)
            # logging.info('*****After {} steps,   loss is {},    accurate is {}'.format(
            #     i,float('%.3f' % loss),float('%.3f' % err))) 
            print('epoch:\t{}\tloss:\t{}\t   error_rate:\t{}'.format(
                i,float('%.5f' % loss),float('%.3f' % err)))
        
        b_err.append(err)
        b_loss.append(loss)
        i += 1

        if not train_or:
            decode = sess.run(Model.decoded,feed_dict=feed)
            print(y[0],'---------',decode_sparse_tensor(decode[0])[0])

    return np.mean(b_err),np.mean(b_loss)

#########################################################
def main(_):

    initializer =tf.truncated_normal_initializer(stddev=0.1)

    with tf.variable_scope('rnn_ctc_model', reuse=None, initializer=initializer):
        train_model = RNNCTC(True)

    with tf.variable_scope('rnn_ctc_model', reuse=True, initializer=initializer):
        eval_model = RNNCTC(False)

    saver = tf.train.Saver(max_to_keep=3)
    base_line = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(6):
            train_data = batch_train_flow(cfg.batch_size)           
            valid_data = batch_valid_flow(cfg.batch_size)

            logging.info('In iteration: {}'.format(i+1))
            train_err,train_loss = run_epoch(sess, train_model,
                        train_data, train_model.train_op, True)
            logging.info('Train Data: loss {}  ,error rate  {}'.format(
                train_loss,train_err
            ))

            valid_err,valid_loss = run_epoch(sess, eval_model,
                        valid_data, tf.no_op(), False)
            logging.info('Valid Data: loss {}  ,error rate  {}'.format(
                valid_loss,valid_err
            ))
            if valid_loss < base_line:
                saver.save(sess,'./save/model-',global_step=i+1)
                base_line = valid_loss
      
if __name__ == '__main__':
    tf.app.run()
