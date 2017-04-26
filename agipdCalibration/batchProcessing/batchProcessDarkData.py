import h5py
import sys
import numpy as np
import time

# fileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/darkData_m2.h5'
# saveFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/python_saved_workspace/darkOffset_agipd11.h5'
fileName = sys.argv[1]
saveFileName = sys.argv[2]

print('\n\n\nstart batchProcessDarkData')
print('fileName = ', fileName)
print('saveFileName = ', saveFileName)
print('')

saveFile = h5py.File(saveFileName, "w", libver='latest')
dset_darkOffset = saveFile.create_dataset("darkOffset", shape=(352, 128, 512), dtype='int16')
dset_darkStandardDeviation = saveFile.create_dataset("darkStandardDeviation", shape=(352, 128, 512), dtype='float')

totalTime = time.time()

print('start loading analog from', fileName)
f = h5py.File(fileName, 'r')
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

print('start saving results at', saveFileName)
dset_darkOffset[...] = means
dset_darkStandardDeviation[...] = standardDeviations
saveFile.flush()
print('saving done')

saveFile.close()

print('batchProcessDarkData took time:  ', time.time() - totalTime, '\n\n')
