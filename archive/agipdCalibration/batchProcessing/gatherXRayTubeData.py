import h5py
import numpy as np
import sys
import time

class GatherXRayTubeData():
    def __init__(self, fileName, safeFileName):

        self.fileName = fileName
        self.saveFileName = safeFileName
        self.dataPathInFile = '/entry/instrument/detector/data'

        print('\n\n\nstart gatherXRayTubeData')
        print('fileName = ', self.fileName)
        print('saveFileName = ', self.saveFileName)
        print('dataPathInFile = ', dataPathInFile)
        print(' ')

        self.run()

    def run(self):

        totalTime = time.time()

        f = h5py.File(self.fileName, 'r')
        dataCount = int(f[dataPathInFile].shape[0] / 2)

        saveFile = h5py.File(self.saveFileName, "w", libver='latest')
        dset_analog = saveFile.create_dataset("analog", shape=(dataCount, 128, 512), dtype='int16')
        dset_digital = saveFile.create_dataset("digital", shape=(dataCount, 128, 512), dtype='int16')

        print('start loading')
        rawData = np.array(f[dataPathInFile])
        print('loading done')
        f.close()

        analog = rawData[::2, ...]
        digital = rawData[1::2, ...]

        print('start saving')
        dset_analog[...] = analog
        dset_digital[...] = digital
        print('saving done')

        saveFile.flush()
        saveFile.close()

        print('gatherXRayTubeData took time:  ', time.time() - totalTime, '\n\n')
