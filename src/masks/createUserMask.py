"""
Create user-defined mask

For example: beam stop, ASIC edges, known bad regions, etc.

combinedMask: logical OR of all sub-masks

Note: when no pixels/cells are masked, plotting throws an exception!

"""

import h5py
import numpy as np
import os

import matplotlib.pyplot as plt
import pyqtgraph as pg


module_id = 'M314'
module_number = 'm7'
temperature = 'temperature_m15C'
base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/{}/{}".format(module_id, temperature)

# Flags for which masks to create
manual = False
systematic = True # memory cell 0
asic_edge = False

# Input files


# Output mask file
save_filename = os.path.join(base_dir, 'cal_output/user_mask_{}_{}_{}.h5'.format(module_id, module_number, temperature))

# Create output file
f_out = h5py.File(save_filename, "w", libver='latest')
dset_combined_mask = f_out.create_dataset("combined_mask", shape=(352, 128, 512), dtype=bool)
masks = f_out.create_group("masks")

combined_mask = np.zeros((352, 128, 512), dtype=bool)


################# manual mask - please edit (e.g. beamstop) ############
manual_mask = np.zeros((128, 512), dtype=bool)
dset_manual_mask = masks.create_dataset("manual_mask", shape=(352, 128, 512), dtype=bool)

if manual:
    # !!Here is an example, edit to reflect your needs/setup!!
    manual_mask[65:105, 200:260] = True
    manual_mask[80:105, 200:511] = True

    manual_mask[80:86, 22:43] = True
    manual_mask[75:100, 31:43] = True

    manual_mask[64:127, 127:195] = True

dset_manual_mask[...] = manual_mask

combined_mask = np.logical_or(manual_mask, combined_mask)
########################################################################


######### systematic mask - please edit ################################
systematic_mask = np.zeros((128, 512, 11, 32), dtype=bool)
dset_systematic_mask = masks.create_dataset("systematic_mask", shape=(352, 128, 512), dtype=bool)

if systematic:
    # systematic_mask[..., 27:29] = True  # 32-column tips
    systematic_mask[..., 0, 0] = True # memory cell 0

systematic_mask = systematic_mask.reshape((128, 512, 352)).transpose(2, 0, 1)
dset_systematic_mask[...] = systematic_mask

combined_mask = np.logical_or(systematic_mask, combined_mask)
########################################################################


######### bad asic borders mask - please edit ##########################
asic_edge_mask = np.zeros((352, 128, 512), dtype=bool)
dset_asic_edge_mask = masks.create_dataset("asic_edge_mask", shape=(352, 128, 512), dtype=bool)

if asic_edge:
    asic_edge_mask[:, (0, 63, 64, 127), :] = True
    for column in np.arange(8):
        asic_edge_mask[:, :, (column * 64, column * 64 + 63)] = True

dset_asic_edge_mask[...] = asic_edge_mask

combined_mask = np.logical_or(asic_edge_mask, combined_mask)
########################################################################


##################### the rest can be left untouched ###################

dset_combined_mask[...] = combined_mask
f_out.flush()
f_out.close()
print('combined_mask saved in ', save_filename)

combined_mask_file = h5py.File(save_filename, "r", libver='latest')
combined_mask = combined_mask_file['combined_mask'][...]
pg.image(combined_mask.transpose(0, 2, 1))

print('\n\npercentage of masked cells: ', 100 * combined_mask.flatten().sum() / combined_mask.size)
print('\n\n\npress enter to quit')
tmp = input()
