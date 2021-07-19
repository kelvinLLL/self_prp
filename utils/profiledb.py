import numpy as np

class ProfileDB:
    def __init__(self, dtype=np.float):
        self.data = {}
        self.dtype = dtype

    def clear_data(self):
        self.data.clear()

    def getProfiles(self, indexes, default=None):
        return [self.data.get(index, default) for index in indexes]

    def setProfiles(self, indexes, profiles):
        for index, profile in zip(indexes, profiles):
            self.data[index] = profile

    def _getWrongProfiles_1_length(self, indexes, default=None):
        key, value = next(iter(self.data.items()))
        return [default if index == key else value for index in indexes]

    def _getWrongProfiles_more_length(self, indexes):
        keys = list(self.data.keys())
        values = list(self.data.values())

        former = np.arange(len(indexes))
        new_key_id = np.arange(len(indexes))
        while True:
            eq = new_key_id == former
            s = eq.sum()
            if s > 0:
                new_key_id[eq] = np.random.randint(
                    low=0, high=len(keys), size=s)
            else:
                break

        return [values[key_id] for key_id in new_key_id]

    def getWrongProfiles(self, indexes, default=None):
        if len(self.data) == 0:
            return [default] * len(indexes)

        if len(self.data) == 1:
            return self._getWrongProfiles_1_length(indexes, default=default)

        return self._getWrongProfiles_more_length(indexes)
