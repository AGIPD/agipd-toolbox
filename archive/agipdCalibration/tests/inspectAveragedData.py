import h5py
import numpy as np

import matplotlib.pyplot as plt
import pyqtgraph as pg

maskFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/agipdCalibration_workspace/mask_m3.h5'
maskFile3 = h5py.File(maskFileName, 'r', libver='latest')
badCellMask3 = maskFile3['/badCellMask'][...]

dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/correctedLyzozymeData/lysozymeData_m3_averageFiltered.h5'

dataFile = h5py.File(dataFileName, 'r', libver='latest')
medianFiltered3 = dataFile['/medianFiltered'][...]  # shape = (3200, 128, 512)
meanFiltered3 = dataFile['/meanFiltered'][...]

# dataFileName = '/gpfs/cfel/fsds/labs/processed/Yaroslav/correctedLyzozymeData/lysozymeData_m4_averageFiltered.h5'
#
# dataFile = h5py.File(dataFileName, 'r', libver='latest')
# medianFiltered4 = dataFile['/medianFiltered'][...]      # shape = (3200, 128, 512)
# meanFiltered4 = dataFile['/meanFiltered'][...]


medianFiltered3_plot = pg.image(medianFiltered3.transpose(0, 2, 1))
meanFiltered3_plot = pg.image(meanFiltered3.transpose(0, 2, 1))

i = 1
