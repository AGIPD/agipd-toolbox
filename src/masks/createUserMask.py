"""
Create user-defined mask; for example: beam stop, ASIC edges, known bad regions, etc.
- inherits from CreateMasks

combinedMask: logical OR of all sub-masks

"""

import h5py
import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import pyqtgraph as pg
from string import Template
# need to tell python where to look for helpers.py
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
SRC_PATH = os.path.join(BASE_PATH, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
# now we can import functions from files in src directory
from helpers import create_dir
from createMaskBase import CreateMasks, read_data



def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir",
                        type=str,
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/processed",
                        help="Directory to get data from")
    parser.add_argument("--outfile_name",
                        type=str,
                        default="mask.h5",
                        help="Filename for output mask file")
    parser.add_argument("--module",
                        type=str,
                        required=True,
                        help="Module ID and position, e.g M310_m3")
    parser.add_argument("--temperature",
                        type=str,
                        required=True,
                        help="temperature to gather, e.g. temperature_30C")
    parser.add_argument("--manual",
                        type=str,
                        default=None,
                        help="Use manually defined mask, e.g. beamstop")
    parser.add_argument("--asic_edge",
                        default=False,
                        action="store_true",
                        help="Mask ASIC edges")
    args = parser.parse_args()

    return args




def create_manual_mask(maskfile, manual_mask):
    """
    Create a manually-defined mask from user input 
    
    Input format:
    memcell, x, y
    
    TODO: also input ranges?
    memcell1:memcell2, x1:x2, y1:y2

    """

    cells = []
    f = open(maskfile, "r")
    for line in f:
        cells.append(line.split(","))
    f.close()

    for c in cells:
        manual_mask[int(c[0]), int(c[1]), int(c[2])] = True

    return manual_mask


def create_asic_edge_mask(asic_edge_mask):
    """
    Mask asic edges

    """
    asic_edge_mask[:, (0, 63, 64, 127), :] = True
    for column in np.arange(8):
        asic_edge_mask[:, :, (column * 64, column * 64 + 63)] = True
    
    return asic_edge_mask


class CreateUserMasks(CreateMasks):

    def __init__(self, base_dir, module, temperature, outfile_name, manual, asic_edge):
        
        CreateMasks.__init__(self, base_dir, module, temperature, outfile_name)

        self.manual = manual
        self.asic_edge = asic_edge
        self.manual_file = None

        if self.manual is not None:
            self.manual_file = os.path.join(self.input_dir, self.manual)




    def run(self):

        # Manually-defined mask
        manual_mask = np.zeros((352, 128, 512), dtype=bool)
        dset_manual_mask = self.masks.create_dataset("manual", shape=(352, 128, 512), dtype=bool)
        if self.manual is not None:
            manual_mask = create_manual_mask(self.manual_file, manual_mask)
        dset_manual_mask[...] = manual_mask
        self.mask_list.append(manual_mask)
        print("Manual % of cells masked: ", 100 * manual_mask.flatten().sum() / manual_mask.size)

        # ASIC edges
        asic_edge_mask = np.zeros((352, 128, 512), dtype=bool)
        dset_asic_edge_mask = self.masks.create_dataset("asic_edges", shape=(352, 128, 512), dtype=bool)
        if self.asic_edge:
            asic_edge_mask = create_asic_edge_mask(asic_edge_mask)
        dset_asic_edge_mask[...] = asic_edge_mask
        self.mask_list.append(asic_edge_mask)
        print("ASIC edges masked, %: ", 100 * asic_edge_mask.flatten().sum() / asic_edge_mask.size)

        # combine all generated masks
        self.combine_masks()
        
        self.f_out.flush()
        self.f_out.close()
        print("\nAll masks saved in ", self.outfile_path)


if __name__ == "__main__":
    
    args = get_arguments()

    base_dir = args.base_dir
    outfile_name = args.outfile_name
    module = args.module
    temperature = args.temperature
    manual = args.manual
    asic_edge = args.asic_edge

    print("Configured parameter:")
    print("base_dir: ", base_dir)
    print("outfile_name: ", outfile_name)
    print("module: ", module)
    print("temperature: ", temperature)
    print("manual: ", manual)
    print("asic_edge: ", asic_edge)


    obj = CreateUserMasks(base_dir, module, temperature, outfile_name, manual, asic_edge)

    obj.run()
