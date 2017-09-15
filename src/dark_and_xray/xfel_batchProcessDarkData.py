import h5py
import sys
import numpy as np
import time


class BatchProcessDarkData():
    def __init__(self, input_fname, output_fname):
        self.input_fname = input_fname
        self.output_fname = output_fname

        self.n_rows = 128
        self.n_cols = 512

        print('\n\n\nstart batchProcessDarkData')
        print('input_fname = ', self.input_fname)
        print('output_fname = ', self.output_fname)
        print('')

        self.run()

    def run(self):

        totalTime = time.time()

        print('start loading analog from', self.input_fname)
        f = h5py.File(self.input_fname, 'r')
        analog = f['analog'][()]
        f.close()
        print('loading done')

        self.n_memcells = analog.shape[1]

        print('start computing means and standard deviations')
        means = np.mean(analog, axis=0)
        # invistigate this (found through trial and error)
        means = np.rollaxis(means, 0, 3)
        standardDeviations = np.empty((128, 512, self.n_memcells))
        for cell in np.arange(self.n_memcells):
            standardDeviations[..., cell] = np.std(analog[:, cell, :, :].astype('float'), axis=0)
        print('done computing means and standard deviations')

        # print('\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!! WATCH OUT!!!! WORKAROUND!!!!! flipping DETECTOR !!!!! !!!!!!!!!!!!!!!!!!!!!!\n\n\n\n\n')
        # means = means[...,::-1] #np.rot90(means, 2)
        # standardDeviations =  standardDeviations[...,::-1] #np.rot90(standardDeviations, 2)

        saveFile = h5py.File(self.output_fname, "w", libver='latest')
        dset_offset = saveFile.create_dataset("offset",
                                              shape=(128, 512, self.n_memcells),
                                              dtype=np.int16)
        dset_standardDeviation = saveFile.create_dataset("standardDeviation",
                                                         shape=(128, 512, self.n_memcells),
                                                         dtype='float')

        print('start saving results at', self.output_fname)
        dset_offset[...] = means
        dset_standardDeviation[...] = standardDeviations
        saveFile.flush()
        print('saving done')

        saveFile.close()

        print('batchProcessDarkData took time:  ', time.time() - totalTime, '\n\n')


if __name__ == "__main__":

    module = "12"

    input_fname = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/kuhnm/R0391-AGIPD{}-gathered.h5".format(module)
    output_fname = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/kuhnm/R0391-AGIPD{}-processed.h5".format(module)

    BatchProcessDarkData(input_fname, output_fname)
