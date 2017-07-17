from __future__ import print_function

import os
import sys
from process_drscs import ProcessDrscs
import argparse
import datetime
import numpy as np

OUTPUT_BASE_DIR = "/gpfs/cfel/fsds/labs/processed/calibration/processed/"
INPUT_BASE_DIR = "/gpfs/cfel/fsds/labs/processed/calibration/processed/"

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Directory to get data from")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Base directory to write results to")
    parser.add_argument("--module",
                        type=str,
                        required=True,
                        help="Module to gather, e.g M310")
    parser.add_argument("--temperature",
                        type=str,
                        required=True,
                        help="temperature to gather, e.g. temperature_30C")
    parser.add_argument("--current",
                        type=str,
                        required=True,
                        help="Current to use, e.g. itestc20")
    parser.add_argument("--asic",
                        type=int,
                        required=True,
                        choices=range(1, 17),
                        help="Asic number")

    return parser.parse_args()

def create_dir(directory_name):
    if not os.path.exists(directory_name):
        try:
            os.makedirs(directory_name)
            print("Dir '{0}' does not exist. Create it.".format(directory_name))
        except IOError:
            if os.path.isdir(directory_name):
                pass


if __name__ == "__main__":

    args = get_arguments()

    module = args.module
    temperature = args.temperature
    current = args.current
    asic = args.asic
    input_base_dir = args.input_dir
    output_base_dir = args.output_dir

    #module = "M314"
    #temperature = "temperature_m15C"
    #current = "itestc150"
    #asic = 1
    #input_base_dir = "/gpfs/cfel/fsds/labs/processed/calibration/processed/"

    print ("configured parameter: ")
    print ("module: ", module)
    print ("temperature: ", temperature)
    print ("current: ", current)
    print ("asic: ", asic)
    print ("input_dir: ", input_base_dir)
    print ("output_dir: ", output_base_dir)

    print ("\nStarted at", str(datetime.datetime.now()))

    rel_file_path = os.path.join(input_base_dir, temperature, "drscs", current)

    input_file_name = "{}_drscs_{}_asic{}.h5".format(module, current, asic)
    input_file_dir = os.path.join(INPUT_BASE_DIR, module, temperature, "drscs", current)
    input_file = os.path.join(input_file_dir, input_file_name)

    output_file_name = "{}_drscs_{}_asic{}_processed.h5".format(module, current, asic)
    output_file_dir = os.path.join(OUTPUT_BASE_DIR, module, temperature, "drscs", current)
    output_file = os.path.join(output_file_dir, output_file_name)

    #create_dir(output_file_dir)

    plot_dir = os.path.join(output_file_dir, "plots")
    create_dir(plot_dir)

    create_error_plots = False
    pixel_v_list = np.arange(64)
    pixel_u_list = np.arange(64)
    mem_cell_list = np.arange(352)

    cal = ProcessDrscs(input_file, plot_dir, create_plots=False)

    for pixel_v in pixel_v_list:
        for pixel_u in pixel_u_list:
            for mem_cell in mem_cell_list:
                pixel = [pixel_v, pixel_u]
                try:
                    cal.run(pixel, mem_cell)
                except KeyboardInterrupt:
                    sys.exit(1)
                except Exception as e:
                    print("Failed to run for pixel {} and mem_cell {}"
                          .format(pixel, mem_cell))
                    print("Error was: {}".format(e))

                    if create_error_plots:
                        try:
                            cal.generate_data_plot()
                        except:
                            print("Failed to generate plot")

                    #raise

    print("\nFinished at", str(datetime.datetime.now()))
