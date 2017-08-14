from __future__ import print_function
import h5py
import numpy as np
import os

raw_file_dir = "/gpfs/cfel/fsds/labs/calibration/current/302-303-314-305/temperature_m15C/drscs/itestc20/"
asic_file_dir = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_m15C/drscs/itestc20/"

module = "M314"
current = "itestc20"

raw_file_list = [
    "M314_m7_drscs_itestc20_col15_00009_part00000.nxs",
    "M314_m7_drscs_itestc20_col26_00009_part00000.nxs",
    "M314_m7_drscs_itestc20_col37_00009_part00000.nxs",
    "M314_m7_drscs_itestc20_col48_00009_part00000.nxs"
]

#       ____ ____ ____ ____ ____ ____ ____ ____
# 0x64 |    |    |    |    |    |    |    |    |
#      | 16 | 15 | 14 | 13 | 12 | 11 | 10 |  9 |
# 1x64 |____|____|____|____|____|____|____|____|
#      |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
# 2x64 |____|____|____|____|____|____|____|____|
#      0*64 1x64 2x64 3x64 4x64 5x64 6x64 7x64 8x64

# index: row_start, row_stop, col_start, col_stop
asic_files = [
    {
        # ASIC 1
        "file_name": "{}_drscs_{}_asic1.h5".format(module, current),
        "index": [1*64, 2*64, 0*64, 1*64],
        "row_position": 1,
    },
    {
        # ASIC 2
        "file_name": "{}_drscs_{}_asic2.h5".format(module, current),
        "index": [1*64, 2*64, 1*64, 2*64],
        "row_position": 1,
    },
    {
        # ASIC 3
        "file_name": "{}_drscs_{}_asic3.h5".format(module, current),
        "index": [1*64, 2*64, 2*64, 3*64],
        "row_position": 1,
    },
    {
        # ASIC 4
        "file_name": "{}_drscs_{}_asic4.h5".format(module, current),
        "index": [1*64, 2*64, 3*64, 4*64],
        "row_position": 1,
    },
    {
        # ASIC 5
        "file_name": "{}_drscs_{}_asic5.h5".format(module, current),
        "index": [1*64, 2*64, 4*64, 5*64],
        "row_position": 1,
    },
    {
        # ASIC 6
        "file_name": "{}_drscs_{}_asic6.h5".format(module, current),
        "index": [1*64, 2*64, 5*64, 6*64],
        "row_position": 1,
    },
    {
        # ASIC 7
        "file_name": "{}_drscs_{}_asic7.h5".format(module, current),
        "index": [1*64, 2*64, 6*64, 7*64],
        "row_position": 1,
    },
    {
        # ASIC 8
        "file_name": "{}_drscs_{}_asic8.h5".format(module, current),
        "index": [1*64, 2*64, 7*64, 8*64],
        "row_position": 1,
    },
    {
        # ASIC 9
        "file_name": "{}_drscs_{}_asic9.h5".format(module, current),
        "index": [0*64, 1*64, 7*64, 8*64],
        "row_position": 0,
    },
    {
        # ASIC 10
        "file_name": "{}_drscs_{}_asic10.h5".format(module, current),
        "index": [0*64, 1*64, 6*64, 7*64],
        "row_position": 0,
    },
    {
        # ASIC 11
        "file_name": "{}_drscs_{}_asic11.h5".format(module, current),
        "index": [0*64, 1*64, 5*64, 6*64],
        "row_position": 0,
    },
    {
        # ASIC 12
        "file_name": "{}_drscs_{}_asic12.h5".format(module, current),
        "index": [0*64, 1*64, 4*64, 5*64],
        "row_position": 0,
    },
    {
        # ASIC 13
        "file_name": "{}_drscs_{}_asic13.h5".format(module, current),
        "index": [0*64, 1*64, 3*64, 4*64],
        "row_position": 0,
    },
    {
        # ASIC 14
        "file_name": "{}_drscs_{}_asic14.h5".format(module, current),
        "index": [0*64, 1*64, 2*64, 3*64],
        "row_position": 0,
    },
    {
        # ASIC 15
        "file_name": "{}_drscs_{}_asic15.h5".format(module, current),
        "index": [0*64, 1*64, 1*64, 2*64],
        "row_position": 0,
    },
    {
        # ASIC 16
        "file_name": "{}_drscs_{}_asic16.h5".format(module, current),
        "index": [0*64, 1*64, 0*64, 1*64],
        "row_position": 0,
    },
]

for i in xrange(len(raw_file_list)):
    file_name = raw_file_list[i]
    raw_file_path = os.path.join(raw_file_dir, file_name)

    print("Reading file: {}".format(file_name))
    raw_file = h5py.File(raw_file_path, "r")

    for asic in asic_files:

        gather_file = h5py.File(os.path.join(asic_file_dir, asic["file_name"]), "r")

        gather_data = gather_file["/entry/instrument/detector/data"][:, :, :, :]

        raw_data = raw_file["/entry/instrument/detector/data"][:, asic["index"][0]:asic["index"][1], asic["index"][2]:asic["index"][3]]

        raw_data.shape = (-1, 352, 2, 64, 64)
        raw_analog = raw_data[:, :, 0, :, :]

        n_charges = raw_data.shape[0]

        # (charges, mem_cells, asic_size, asic_size)
        # is transposed to
        # (asic_size, asic_size, mem_cells, charges)
        transpose_order = (2,3,1,0)

        raw_analog_transposed = raw_analog.transpose(transpose_order)

        if asic["row_position"] == 0:
            col_index = np.arange(3 - i, 64, 4)
        else:
            col_index = np.arange(i, 64, 4)

        if np.all(raw_analog_transposed[:, col_index, :, :n_charges] == gather_data[:, col_index, :, :n_charges]):
            print(asic["file_name"], "correct")
        else:
            print(asic["file_name"], "not correct")

        gather_file.close()
        raw_file.close()
