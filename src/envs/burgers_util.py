import numpy as np
from phi.flow import struct
from phi.physics.field import AnalyticField


@struct.definition()
class GaussianClash(AnalyticField):

    def __init__(self, batch_size, leftloc=None, leftamp=None, leftsig=None, rightloc=None, rightamp=None, rightsig=None):
        AnalyticField.__init__(self, rank=1)
        self.batch_size = batch_size
        if None in [leftloc, leftamp, leftsig, rightloc, rightamp, rightsig]:
            self.leftloc = np.random.uniform(0.2, 0.4, self.batch_size)
            self.leftamp = np.random.uniform(0, 3, self.batch_size)
            self.leftsig = np.random.uniform(0.05, 0.15, self.batch_size)
            self.rightloc = np.random.uniform(0.6, 0.8, self.batch_size)
            self.rightamp = np.random.uniform(-3, 0, self.batch_size)
            self.rightsig = np.random.uniform(0.05, 0.15, self.batch_size)
        else:
            self.leftloc = leftloc*np.ones(self.batch_size)
            self.leftamp = leftamp*np.ones(self.batch_size)
            self.leftsig = leftsig*np.ones(self.batch_size)
            self.rightloc = rightloc*np.ones(self.batch_size)
            self.rightamp = rightamp*np.ones(self.batch_size)
            self.rightsig = rightsig*np.ones(self.batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        left = self.leftamp * np.exp(-0.5 * (idx - self.leftloc) ** 2 / self.leftsig ** 2)
        right = self.rightamp * np.exp(-0.5 * (idx - self.rightloc) ** 2 / self.rightsig ** 2)
        result = left + right
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data


@struct.definition()
class GaussianForce(AnalyticField):
    def __init__(self, batch_size, loc=None, amp=None, sig=None):
        AnalyticField.__init__(self, rank=1)
        if None in [loc, amp, sig]:
            self.loc = np.random.uniform(0.4, 0.6, batch_size)
            self.amp = np.random.uniform(-0.05, 0.05, batch_size) * 32
            self.sig = np.random.uniform(0.1, 0.4, batch_size)
        else:
            self.loc = loc*np.ones(batch_size)
            self.amp = amp*np.ones(batch_size)
            self.sig = sig*np.ones(batch_size)

    def sample_at(self, idx, collapse_dimensions=True):
        idx = np.swapaxes(idx, 0, -1)  # batch last to match random values
        result = self.amp * np.exp(-0.5 * (idx - self.loc) ** 2 / self.sig ** 2)
        result = np.swapaxes(result, 0, -1)
        return result

    @struct.constant()
    def data(self, data):
        return data