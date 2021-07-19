from typing import List
import numpy as np
import math


class OneDiffDataLoader:
    def __init__(self, *x: List[np.ndarray], batch: int = 1, shuffle: bool = True, one_diff_index=None):
        assert one_diff_index is not None
        x0_length = x[0].shape[0]
        assert all([x0_length == xi.shape[0] for xi in x[1:]])
        self.x = x
        self.batch = batch
        self.shuffle = shuffle
        self._len_element = x0_length
        self._len = math.ceil(x0_length / batch)
        self.diff_index = one_diff_index

    def __iter__(self):
        return self.DataLoaderIter(*self.x, diff_index=self.diff_index, batch=self.batch, shuffle=self.shuffle)

    def __len__(self):
        return self._len

    def len_element(self):
        return self._len_element

    class DataLoaderIter:
        def __init__(self, *x: List[np.ndarray], batch: int = 1, shuffle: bool = True, diff_index=None, **kwargs):
            self.x = x
            self.batch = batch
            self.indexes = np.arange(x[0].shape[0], dtype=np.int32)
            if shuffle:
                np.random.shuffle(self.indexes)
            self.index = 0
            self.diff_index = diff_index

        def __next__(self):
            length = len(self.indexes)
            if self.index >= length:
                raise StopIteration()

            index_r = self.index + self.batch
            if index_r > length:
                index_r = length

            index = self.indexes[self.index:index_r]

            flag = False
            diff_y = self.diff_index
            for i, yy in enumerate(self.x[1][index], self.index):
                if yy == diff_y:
                    if flag:
                        index_r = i
                        break
                    else:
                        flag = True
            index = self.indexes[self.index:index_r]
            self.index = index_r
            return (x[index] for x in self.x)
