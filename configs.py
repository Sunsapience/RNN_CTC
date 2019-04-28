from easydict import EasyDict as edict

__C = edict()
cfg = __C 

#######################################################
__C.input_size = 28

__C.class_size = 10+1

__C.hidden_size = 160

__C.max_time = 28*10

__C.learning_rate = 0.005

__C.sequence_len = 10

__C.batch_size = 100