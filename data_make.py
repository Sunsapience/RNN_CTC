import numpy as np 
import matplotlib.pyplot as plt
import pylab
import pickle

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("D:/tmp/data/", one_hot=True)

def make_seq_data(number,len_seq,train=True):
    data_x = []
    data_y = []
    if train:
        for i in range(number):
            images, labels = mnist.train.next_batch(len_seq)
            images = np.reshape(images,(-1,28,28))
            z = np.zeros((28,28*len_seq))
            for j in range(len_seq):
                z[:,j*28:(j+1)*28]=images[j]
            data_x.append(z)
            data_y.append([np.argmax(k) for k in labels])
    else:
        assert number*len_seq<=10000
        images = mnist.test.images[:number*len_seq].reshape((-1, 28, 28))
        labels = mnist.test.labels[:number*len_seq]
        for i in range(number):
            z = np.zeros((28,28*len_seq))
            for j in range(len_seq):
                z[:,j*28:(j+1)*28]=images[i*len_seq+j]   
            data_x.append(z)
            data_y.append([np.argmax(k) for k in labels[i*len_seq:(i+1)*len_seq]])

    return data_x,data_y

def save_data(data,test_or_train='test'):
    if test_or_train == 'test':
        name = 'test'
    elif test_or_train == 'valid':
        name = 'valid'
    else:
        name = 'train'
    with open('./data/'+name+'_data.pkl','wb') as f:
        pickle.dump(data,f)

if __name__ == '__main__':
    train_data = make_seq_data(8000,10,train=True)
    valid_data = make_seq_data(1000,10,train=True)
    test_data = make_seq_data(1000,10,train=False)
    save_data(test_data,test_or_train='test')
    save_data(valid_data,test_or_train='valid')
    save_data(train_data,test_or_train='train')

        
# images = mnist.test.images[:10].reshape((-1, 28, 28))
# labels = mnist.test.labels[:10]
# yy = [np.argmax(k) for k in labels[:10]]
# print(yy)
# # images, labels = mnist.train.next_batch(10)
# # images = np.reshape(images,(-1,28,28))

# z= np.zeros((28,280))
# for i in range(10):
#     z[:,i*28:(i+1)*28]=images[i]
# # print([np.argmax(i) for i in labels])
# plt.imshow(z)
# pylab.show()


