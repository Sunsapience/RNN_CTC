import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] =False

import random
import pylab
import tensorflow as tf
import numpy as np

from configs import cfg
from data_utils import batch_test_flow, decode_sparse_tensor
from model import RNNCTC


with tf.variable_scope('rnn_ctc_model', reuse=None):
    test_model = RNNCTC(False)

test_data = batch_test_flow(cfg.batch_size)

####################################
def run_epoch(sess,Model=test_model,data_queue=test_data):
    re_0 = []
    re_1 = []
    re_2 = []
    for x,y,sparse_y in data_queue:
        feed = {
            Model.inputs: x,
            Model.labels: sparse_y,
            Model.seq_len: [cfg.max_time]*len(x)
        }
        decode = sess.run(Model.decoded, feed_dict=feed)

        re_0 += np.transpose(x,(0,2,1)).tolist() 
        re_1 += y
        re_2 += decode_sparse_tensor(decode[0])

    return re_0,re_1,re_2


def main(_):
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,'./save/model--6') 
        result = run_epoch(sess)

        for i in [random.randint(0,len(result[1])) for _ in range(5)]:
            plt.figure()
            plt.title("标签:  {}\n 预测:  {},".format(result[1][i],result[2][i]))
            plt.imshow(result[0][i])
        plt.show()

if __name__ == '__main__':
    tf.app.run()
