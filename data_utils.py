'''
the function of sparse_tuple_from and decode_sparse_tensor refer:https://github.com/stardut/ctc-ocr-tensorflow
'''

import pickle ,random
import numpy as np 
from configs import cfg

f = open('./data/train_data.pkl','rb')
Data = pickle.load(f)
f.close()
train_x,train_y = Data[0],Data[1]

f = open('./data/test_data.pkl','rb')
Data = pickle.load(f)
f.close()
test_x,test_y = Data[0],Data[1]

f = open('./data/valid_data.pkl','rb')
Data = pickle.load(f)
f.close()
valid_x,valid_y = Data[0],Data[1]

def batch_train_flow(batch_size):
    n_batches = len(train_x) // batch_size
    o = list(range(n_batches * batch_size))
    random.shuffle(o)

    x = [train_x[oo] for oo in o]
    y = [train_y[oo] for oo in o]

    for j in range(n_batches):
        output_x = x[j*batch_size:(j+1)*batch_size] #[batch_size,28,max_time]
        output_y = y[j*batch_size:(j+1)*batch_size]
        
        yield np.transpose(output_x,(0,2,1)), output_y, sparse_tuple_from(output_y)

def batch_valid_flow(batch_size):

    x = valid_x
    y = valid_y
    n_batches = len(valid_x) // batch_size
    for j in range(n_batches):
        output_x = x[j*batch_size:(j+1)*batch_size]
        output_y = y[j*batch_size:(j+1)*batch_size]
        
        yield np.transpose(output_x,(0,2,1)), output_y, sparse_tuple_from(output_y)

def batch_test_flow(batch_size):

    x = test_x
    y = test_y
    n_batches = len(test_x) // batch_size
    for j in range(n_batches):
        output_x = x[j*batch_size:(j+1)*batch_size]
        output_y = y[j*batch_size:(j+1)*batch_size]
        
        yield np.transpose(output_x,(0,2,1)), output_y, sparse_tuple_from(output_y)


def sparse_tuple_from(sequences, dtype=np.int32):

    indices = []
    values = []        
    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64) 
   
    return indices, values, shape

def decode_sparse_tensor(sparse_tensor):
    """Transform sparse to sequences ids."""
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)

    result = []
    for index in decoded_indexes:
        ids = [sparse_tensor[1][m] for m in index]
        result.append(ids)
    return result
