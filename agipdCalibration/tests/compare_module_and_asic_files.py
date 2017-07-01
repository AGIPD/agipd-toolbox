from __future__ import print_function
import h5py
import numpy as np
import os

module_file_path = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_40C/drscs/itestc150/"
axis_file_path = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M314/temperature_40C/drscs/itestc150/"

module = "M314"
current = "itestc150"

g = h5py.File(os.path.join(module_file_path, "{}_drscs_{}_chunked.h5".format(module, current)), "r")

# index: col_start, col_stop, row_start, row_stop
asic_files = [
    {
        # ASIC 1
        "file_name": "{}_drscs_{}_asic1.h5".format(module, current),
        "index": [0*64, 1*64, 0*64, 1*64]
    },
    {
        # ASIC 2
        "file_name": "{}_drscs_{}_asic2.h5".format(module, current),
        "index": [0*64, 1*64, 1*64, 2*64]
    },
    {
        # ASIC 3
        "file_name": "{}_drscs_{}_asic3.h5".format(module, current),
        "index": [0*64, 1*64, 2*64, 3*64]
    },
    {
        # ASIC 4
        "file_name": "{}_drscs_{}_asic4.h5".format(module, current),
        "index": [0*64, 1*64, 3*64, 4*64]
    },
    {
        # ASIC 5
        "file_name": "{}_drscs_{}_asic5.h5".format(module, current),
        "index": [0*64, 1*64, 4*64, 5*64]
    },
    {
        # ASIC 6
        "file_name": "{}_drscs_{}_asic6.h5".format(module, current),
        "index": [0*64, 1*64, 5*64, 6*64]
    },
    {
        # ASIC 7
        "file_name": "{}_drscs_{}_asic7.h5".format(module, current),
        "index": [0*64, 1*64, 6*64, 7*64]
    },
    {
        # ASIC 8
        "file_name": "{}_drscs_{}_asic8.h5".format(module, current),
        "index": [0*64, 1*64, 7*64, 8*64]
    },
    {
        # ASIC 9
        "file_name": "{}_drscs_{}_asic9.h5".format(module, current),
        "index": [1*64, 2*64, 0*64, 1*64]
    },
    {
        # ASIC 10
        "file_name": "{}_drscs_{}_asic10.h5".format(module, current),
        "index": [1*64, 2*64, 1*64, 2*64]
    },
    {
        # ASIC 11
        "file_name": "{}_drscs_{}_asic11.h5".format(module, current),
        "index": [1*64, 2*64, 2*64, 3*64]
    },
    {
        # ASIC 12
        "file_name": "{}_drscs_{}_asic12.h5".format(module, current),
        "index": [1*64, 2*64, 3*64, 4*64]
    },
    {
        # ASIC 13
        "file_name": "{}_drscs_{}_asic13.h5".format(module, current),
        "index": [1*64, 2*64, 4*64, 5*64]
    },
    {
        # ASIC 14
        "file_name": "{}_drscs_{}_asic14.h5".format(module, current),
        "index": [1*64, 2*64, 5*64, 6*64]
    },
    {
        # ASIC 15
        "file_name": "{}_drscs_{}_asic15.h5".format(module, current),
        "index": [1*64, 2*64, 6*64, 7*64]
    },
    {
        # ASIC 16
        "file_name": "{}_drscs_{}_asic16.h5".format(module, current),
        "index": [1*64, 2*64, 7*64, 8*64]
    },
]

for asic in asic_files:

    f = h5py.File(os.path.join(axis_file_path, asic["file_name"]), "r")

    dataf = f["/analog"][:, :, :, :]
    datag = g["/analog"][:, :, asic["index"][0]:asic["index"][1], asic["index"][2]:asic["index"][3]]

    if np.all(dataf == datag):
        print(asic["file_name"], "correct")
    else:
        print(asic["file_name"], "not correct")

    f.close()
