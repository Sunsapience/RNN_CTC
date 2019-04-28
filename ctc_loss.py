# @author: Wang LG
# @file: ctc_loss       
# @time: 19-4-22 8:02 PM

import tensorflow as tf 
import numpy as np

class CTC():
    def __init__(self,inputs,labels,L):
        self.inputs = inputs # rnn/cnn经过 softmax 层后的输出，格式[batch_size, max_time, num_classes] ,概率
                             # num_classes = num_label + 1 ,多出一个为空白blank
         # 标签，格式[batch_size,2*num_label+1] 添加blank后地标签
        self.L = L           # num_label
        self.T = inputs.get_shape()[1]  # max_time
        self.B = inputs.get_shape()[0]  # batch_size

        self.labels = labels # 标签，格式[batch_size,2*num_label+1] 添加blank后地标签
        self.alpha,self.log_p = self.forward_and_like_prob() # 前向概率矩阵和似然概率
        self.ctc_loss = -self.log_p  #CTC损失

    def forward_and_like_prob(self):
        alpha = tf.py_func(forword,[self.inputs,self.labels,self.B,self.T,self.L],tf.float32)
        log_p = [tf.log(alpha[k,-1,-1] + alpha[k,-1,-2]) for k in range(self.B)]
        return alpha,tf.convert_to_tensor(log_p)

    def greedy_decode(self):
        blank_result = tf.argmax(self.inputs, axis=-1)
        last_result = tf.py_func(remove_blank,[blank_result,self.L],tf.int32)
        return last_result

    def beam_decode(self):  
        last_result = tf.py_func(beam_search,[self.inputs,self.B,self.T,self.L],tf.int32)
        return last_result


def forword(y,labels,B,T,L):
    alpha = np.zeros([B,T,2*L+1])

    for k in  range(B):
        alpha[k,0,0] = y[k, 0,labels[k,0]]
        alpha[k,0,1] = y[k, 0,labels[k,1]]

        for t in range(1,T):
            for i in range(L*2+1):
                s = labels[k,i]
                a = alpha[k,t - 1,i]
 
                if i - 1 >= 0:
                    a += alpha[k,t - 1, i - 1]
                if i - 2 >= 0 and s != L and s != labels[k,i - 2]:
                    a += alpha[k,t - 1, i - 2]

                alpha[k,t, i] = a * y[k,t, s]
    return alpha

def remove_blank(labels,blank):

    new_labels = []

    for label in labels:
        new_label = []
        previous = None

        for l in label:
            if l != previous:
                new_label.append(l)
                previous = l

        # remove blank     
        new_label = [l for l in new_label if l != blank]
        
        new_labels += [new_label]
    
    # 由于无法将不等长序列转化成tensorflow的张量，故添加了对齐标签，去掉对齐标签后可显示不等长长度
    # 0---(L-1)为正常标签，L 为 blank 标签， L+1 为对齐标签
    L = blank+1  #将第L+1 作为对齐标签
    N = L*np.ones((len(labels),L),np.int32) 
    for k, row in enumerate(new_labels):
        N[k,:len(row)] =  row
    return N

def beam_search(y,B,T,L,beam_size=3):
    V = L+1
    log_y = np.log(y)
    batch_beam = []
    for k in range(B):
        beam = [([], 0)]
        for t in range(T): 
            new_beam = []
            for prefix, score in beam:
                for i in range(V):  
                    new_prefix = prefix + [i] # 加入新的状态
                    new_score = score + log_y[k, t, i] # log(p1*p2)=log(p1)+log(p2)

                    new_beam.append((new_prefix, new_score))

            new_beam.sort(key=lambda x: x[1], reverse=True) 
            beam = new_beam[:beam_size] # 选择合适状态
        batch_beam  += [beam]

    beam_result_ = [[batch_beam[k][i][0] for i in range(beam_size)] for k in range(B)]
    beam_result = np.zeros((B,beam_size,V),np.int32)

    for i in range(B):
        re = remove_blank(beam_result_[i],L)
        beam_result[i] = re
    return beam_result
    # N = V*np.ones((B,beam_size,V),np.int32)
    # for k in range(B):
    #     for i, row in enumerate(batch_beam[k]):
    #         N[k,i,:len(row[0])] =  row[0]
    #         #N[k,i,-1]=np.exp(row[1])
    # return N

# 去掉对齐标签
def remove_align(labels,align,greed_or_beam):
    # align 为对齐标签
    labels_ = labels.tolist()
    if greed_or_beam == 'greed':
        result = [[l for l in label if l != align] for label in labels_]
    else:
        result = [[[l for l in label if l != align] for label in batch] for batch in labels_]

    return result 
'''
def insert_blank(labels, blank,B,L):
    new_labels = [[blank for i in range(2*L+1)] for k in  range(B)]
    for k in range(B):
        for j in range(len(labels[k])):
            new_labels[k][2*j+1] = labels[k,j]
    return np.array(new_labels)



def forward_and_like_prob(self):
    blank = tf.constant(self.L)
    zero = tf.constant(0.0)
    alpha = [[[zero for i in range(self.L*2 + 1)] for t in  range(self.T)] for k in range(self.B)]
    log_p = []
    for k in range(self.B):
        alpha[k][0][0] = self.inputs[k, 0,self.labels[k,0]]
        alpha[k][0][1] = self.inputs[k, 0,self.labels[k,1]]

        for t in range(1,self.T):
            for i in range(self.L*2+1):
                s = self.labels[k,i]
                a = alpha[k][t - 1][i]

                if  i - 1 >= 0:
                    a += alpha[k][t - 1][i - 1]
                if  i - 2 >= 0:
                    r_1,r_2 = tf.not_equal(s,blank),tf.not_equal(s,self.labels[k,i - 2])
                    r_3 = tf.cond(r_1,lambda: r_2, lambda: False)
                    result = tf.cond(r_3,lambda: alpha[k][t - 1][i - 2], lambda: tf.constant(0))
                    a += result

                alpha[k][t][i] = a * self.inputs[k,t,s]

        log_p.append(tf.log(alpha[k][-1][-1] + alpha[k][-1][-2]))

    return tf.convert_to_tensor(alpha),tf.convert_to_tensor(log_p)
'''   
                    


