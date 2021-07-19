import math
import numpy as np
import pandas as pd

from typing import Dict

def forced_reshape_1(array, *args):
    '''
    return array.reshape(-1,*args)
    '''
    batch = 1
    for arg in args:
        batch *= arg
    
    r = array.size % batch
    if r > 0:
        array = array.reshape(-1)[:-r]
    return array.reshape(-1, *args)

def sequence_drop(array, seq):
    '''
    array.shape == (n, *shape)
    return shape == (n', seq, *shape)
    '''
    r = array.shape[0] % seq
    if r > 0:
        array = array[: - r]
    return array.reshape(-1, seq, *array.shape[1:])

def tranpose_seq_batch(array:np.ndarray):
    if len(array.shape) <=1:
        return array
    axis = list(range(len(array.shape)))
    axis[0] = 1
    axis[1] = 0
    return np.transpose(array,axis)

def index_by(y: np.ndarray):
    index, pos_l = np.unique(y, return_index=True)
    pos_r = np.zeros_like(pos_l)
    pos_r[:-1] = pos_l[1:]
    pos_r[-1] = len(y)
    index = dict(zip(index, zip(pos_l, pos_r)))

    return index


class DataLoader:
    '''
    sequence is int -> Train: [Sequence, batch, features*]
    sequence is None -> Eval: [batch, features*]
    '''
    def __init__(self, x: np.ndarray, y: np.ndarray, sequence: int, batch: int = 32):
        assert len(x) == len(y)

        self.sequence = sequence
        self.batch = batch

        arg = y.argsort()
        x = x[arg]
        y = y[arg]
        index = index_by(y)
        self.x = x
        self.y = y
        self.index: Dict[np.ndarray, np.ndarray] = index

        cnt = sum([(pos_r-pos_l) for pos_l, pos_r in self.index.values()])

        self._len = int(math.ceil(cnt / batch))


    def __iter__(self):
        return self.DataLoaderTrainIter(self.x, self.index, self.sequence, self.batch)

    def __len__(self):
        return self._len

    def len_element(self):
        return len(self.x)

    class DataLoaderTrainIter:
        def __init__(self, x, index, sequence, batch):
            self.batch = batch

            x, y, x_ = self.sequence(x, index,  sequence)

            index = np.random.permutation(np.arange(x.shape[0]))

            self.x = x[index]
            self.y = y[index]
            self.x_ = x_[index]

            self.index = 0

        @classmethod
        def sequence(cls, x, index, sequence):
            xx = []
            xx_ = []
            yy = []
            for key, (pos_l,pos_r) in index.items():
                data = x[pos_l:pos_r]

                x_index = np.random.randint(0, pos_r-pos_l, (data.shape[0], sequence))
                x_index_ = np.random.randint(
                    0, x.shape[0] - (pos_r-pos_l), data.shape[0])
                x_index_[x_index_ >= pos_l] += pos_r-pos_l

                xxx = data[x_index]
                xxx_ = x[x_index_]

                xx.append(xxx)
                xx_.append(xxx_)
                yy.extend([key]*x_index.shape[0])

            xx = np.concatenate(xx)
            yy = np.array(yy)
            xx_ = np.concatenate(xx_)

            return xx, yy, xx_
            
        def __next__(self):
            length = len(self.x)
            if self.index > length:
                raise StopIteration()

            right = self.index + self.batch
            if right > length:
                right = length
            index = self.index
            self.index += self.batch

            return self.x[index:right], self.y[index:right], self.x_[index:right]
