"""
Base class to create masks
- CreateCalMasks and CreateUserMasks inherit from here

combined_mask: overall mask (logical OR of all sub-masks)

"""

import h5py
import numpy as np
import os
import sys
# need to tell python where to look for helpers.py
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#print(BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# now we can import functions from files in src directory
from helpers import create_dir


def read_data(filename, dataset_list):
    
    f = h5py.File(filename, 'r')
    dset_list = []
    for d in dataset_list:
        dset = f[d][...]
        dset_list.append(dset)

    f.close()

    return dset_list


class CreateMasks():
    
    def __init__(self, base_dir, module, temperature, outfile_name):
        

        self.base_dir = base_dir
        self.module = module
        self.module_id = module.split("_")[0]
        self.module_number = module.split("_")[1]
        
        self.temperature = temperature


        self.asic_mapping = [[16, 15, 14, 13, 12, 11, 10, 9],
                             [1,   2,  3,  4,  5,  6,  7, 8]]
        
        self.combined_mask = np.zeros((352, 128, 512), dtype=bool)
        self.mask_list = []
        
        self.input_dir = os.path.join(self.base_dir, self.module_id, self.temperature)  
        self.output_dir = os.path.join(self.input_dir, "cal_output")
        self.plot_dir = os.path.join(self.output_dir, "plots")

        self.outfile_name = outfile_name

        create_dir(self.output_dir)
        create_dir(self.plot_dir)
        self.outfile_path = os.path.join(self.output_dir, self.outfile_name)
        self.f_out = h5py.File(self.outfile_path, "w", libver='latest')
        self.dset_combined_mask = self.f_out.create_dataset("combined_mask", shape=self.combined_mask.shape, dtype=bool)
        self.masks = self.f_out.create_group("masks")
        

    def combine_masks(self):

        # combine all generated masks
        for m in self.mask_list:
            self.combined_mask = np.logical_or(self.combined_mask, m)
            
        self.dset_combined_mask[...] = self.combined_mask
        print('\n\nTotal percentage of masked cells: ', 100 * self.combined_mask.flatten().sum() / self.combined_mask.size, '\n')
