import h5py
import sys
import numpy as np
import time


class BatchProcessDarkData():
    def __init__(self, fileName, safeFileName):
        self.fileName = fileName
        self.saveFileName = saveFileName

        print('\n\n\nstart batchProcessDarkData')
        print('fileName = ', self.fileName)
        print('saveFileName = ', self.saveFileName)
        print('')

        self.run()

    def run(self):

        saveFile = h5py.File(self.saveFileName, "w", libver='latest')
        dset_darkOffset = saveFile.create_dataset("darkOffset", shape=(352, 128, 512), dtype='int16')
        dset_darkStandardDeviation = saveFile.create_dataset("darkStandardDeviation", shape=(352, 128, 512), dtype='float')

        totalTime = time.time()

        print('start loading analog from', self.fileName)
        f = h5py.File(self.fileName, 'r')
        analog = f['analog'][()]
        f.close()
        print('loading done')

        print('start computing means and standard deviations')
        means = np.mean(analog, axis=0)
        standardDeviations = np.empty((352, 128, 512))
        for cell in np.arange(352):
            standardDeviations[cell, ...] = np.std(analog[:, cell, :, :].astype('float'), axis=0)
        print('done computing means and standard deviations')

        # print('\n\n\n\n\n!!!!!!!!!!!!!!!!!!!!! WATCH OUT!!!! WORKAROUND!!!!! flipping DETECTOR !!!!! !!!!!!!!!!!!!!!!!!!!!!\n\n\n\n\n')
        # means = means[...,::-1] #np.rot90(means, 2)
        # standardDeviations =  standardDeviations[...,::-1] #np.rot90(standardDeviations, 2)

        print('start saving results at', self.saveFileName)
        dset_darkOffset[...] = means
        dset_darkStandardDeviation[...] = standardDeviations
        saveFile.flush()
        print('saving done')

        saveFile.close()

        print('batchProcessDarkData took time:  ', time.time() - totalTime, '\n\n')
