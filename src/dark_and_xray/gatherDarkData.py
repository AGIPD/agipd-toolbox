import h5py
import sys
import numpy as np
import time

class GatherDarkData():
    def __init__(self, nParts, fileName, saveFileName):

        self.nParts = nParts
        self.fileName = fileName
        sel.saveFileName = saveFileName
        self.dataPathInFile = '/entry/instrument/detector/data'

        print('\n\n\nstart gatherDarkData')
        print('fileName = ', self.fileName)
        print('saveFileName = ', self.saveFileName)
        print('')

        self.run()

    def run(self):

        totalTime = time.time()

        # Data split into several files - "parts"
        # first calculate how many total events
        # assumption - same number of events in each file!
        print('start loading', self.dataPathInFile, 'from', self.fileName)
        f = h5py.File(self.fileName + '00.nxs', 'r')
        dataCountPerFile = int(f[self.dataPathInFile].shape[0] / 2 / 352) #how many per file
        dataCount = dataCountPerFile * self.nParts #times nParts files for total
        f.close()

        analog = np.zeros((dataCount, 352, 128, 512), dtype='int16')
        digital = np.zeros((dataCount, 352, 128, 512), dtype='int16')

        # Loop over all nParts files, read in data
        for j in np.arange(self.nParts):
            if j <= 9:
                fDark_j = self.fileName + '0' + str(j) + '.nxs'
            else:
                fDark_j = self.fileName + str(j) + '.nxs'
            print('start loading ', self.dataPathInFile, ' from ', fDark_j)
            f = h5py.File(fDark_j, 'r')
            rawData = f[self.dataPathInFile][()]
            print('loading done')

            print('start reshaping')
            rawData.shape = (dataCountPerFile, 352, 2, 128, 512)
            analog[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, :, :] = rawData[:, :, 0, :, :]
            digital[j * dataCountPerFile:(j + 1) * dataCountPerFile, :, :, :] = rawData[:, :, 1, :, :]
            print('finished reshaping')

            f.close()
            print('finished loading all data')

        #analog = rawData[::2, ...]
        #analog.shape = (-1, 352, 128, 512)
        #digital = rawData[1::2, ...]
        #digital.shape = (-1, 352, 128, 512)

        saveFile = h5py.File(self.saveFileName, "w", libver='latest')
        dset_analog = saveFile.create_dataset("analog", shape=analog.shape, compression=None, dtype='int16')
        dset_digital = saveFile.create_dataset("digital", shape=digital.shape, compression=None, dtype='int16')

        print('start saving to ', self.saveFileName)
        dset_analog[...] = analog
        dset_digital[...] = digital
        saveFile.close()
        print('saving done')

        print('gatherDarkData took time:  ', time.time() - totalTime, '\n\n')

