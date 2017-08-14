import h5py
import numpy as np
import sys

fileNameRoot = '/asap3/petra3/gpfs/p11/2016/data/11001485/raw/m' + sys.argv[1] + '_Lysozyme_Yb_00200_phi_'
dataPathInFile = '/entry/instrument/detector/data'

saveFileName = '/gpfs/cfel/cxi/scratch/user/gevorkov/python_saved_workspace/lysozymeData_m' + sys.argv[1] + '_calibrationChunked.h5'

analog = np.array([])
digital = np.array([])

analogList = []
digitalList = []

availableAngles = np.arange(160)
availableSubAngles = np.arange(0, 1000, 50)

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_analog = saveFile.create_dataset("analog", shape=(0, 352, 128, 512), chunks=(100, 352, 64, 64), maxshape=(None, 352, 128, 512), dtype='int16')
dset_digital = saveFile.create_dataset("digital", shape=(0, 352, 128, 512), chunks=(100, 352, 64, 64), maxshape=(None, 352, 128, 512), dtype='int16')

for i in availableAngles:
    print('current angle:', i)

    for j in availableSubAngles:
        fileName = fileNameRoot + str(i).zfill(4) + '.' + str(j).zfill(3) + '_00000.nxs'

        f = h5py.File(fileName, 'r')
        rawData = f[dataPathInFile][...]
        f.close()

        analog_new = rawData[::2, ...]
        digital_new = rawData[1::2, ...]

        analog_new.shape = (1,) + analog_new.shape

        digital_new.shape = (1,) + digital_new.shape

        analogList.append(analog_new)
        digitalList.append(digital_new)

        if len(analogList) == 100 or (i == availableAngles[-1] and j == availableSubAngles[-1]):
            print('saving')
            analog_tmp = np.concatenate(tuple(analogList), axis=0)
            digital_tmp = np.concatenate(tuple(digitalList), axis=0)

            analogList = []
            digitalList = []

            dset_analog.resize(dset_analog.shape[0] + analog_tmp.shape[0], axis=0)
            dset_digital.resize(dset_digital.shape[0] + digital_tmp.shape[0], axis=0)

            dset_analog[-analog_tmp.shape[0]:, ...] = analog_tmp
            dset_digital[-digital_tmp.shape[0]:, ...] = digital_tmp
            saveFile.flush()

saveFile.close()
