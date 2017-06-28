from __future__ import print_function

import os
import sys
from gatherdata import GatherData
import argparse
import datetime


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--module",
                        type=str,
                        required=True,
                        help="Module to gather, e.g M310_m7")
    parser.add_argument("--temperature",
                        type=str,
                        required=True,
                        help="temperature to gather, e.g. temperature_30C")
    parser.add_argument("--current",
                        type=str,
                        required=True,
                        help="Current to use, e.g. itestc20")
    parser.add_argument("--column_spec",
                        type=int,
                        nargs='+',
                        required=True,
                        help="which index files to use for which column, e.g.9, 10, 11, 12")

    args = parser.parse_args()

    if len(args.column_spec) != 4:
        print("The have to be columns 4 defined")
        sys.exit(1)

    return args


if __name__ == "__main__":

    INPUT_BASE_PATH = "311-312-301-300-310-234"
    OUTPUT_BASE_PATH = "/gpfs/cfel/fsds/labs/processed/calibration/processed/"

    args = get_arguments()

    #module = "M310_m7"
    #temperature = "temperature_m20C"
    #current = "itestc20"
    module = args.module
    temperature = args.temperature
    current = args.current

    # [[<column>, <file index>],...]
    # e.g. for a file name of the form M234_m8_drscs_itestc150_col15_00001_part00000.nxs
    # the entry would be                                         [15,   1]
    #column_specs = [[15, 9], [26, 10], [37, 11], [48, 12]]
    column_specs = [[15, args.column_spec[0]],
                    [26, args.column_spec[1]],
                    [37, args.column_spec[2]],
                    [48, args.column_spec[3]]]

    max_part = False
    #max_part = 10

    print ("configured parameter: ")
    print ("module: ", module)
    print ("temperature: ", temperature)
    print ("current: ", current)
    print ("column_specs: ", column_specs)
    print ("max_part: ", max_part)
    print ("\nStarted at", str(datetime.datetime.now()))

    #rel_file_path = "311-312-301-300-310-234/temperature_m20C/drscs/itestc20"
    rel_file_path = os.path.join(INPUT_BASE_PATH, temperature, "drscs", current)

    file_base_name = "{}_drscs_{}".format(module, current)
    #file_base_name = "M301_m3_drscs_itestc150"

    output_file_name = "{}_chunked.h5".format(file_base_name)
    #output_file_path = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M310/temperature_30C/drscs/itestc20"
    module_split = module.split("_")
    output_file_path = os.path.join(OUTPUT_BASE_PATH, module_split[0], temperature, "drscs", current)
    output_file = os.path.join(output_file_path, output_file_name)

    GatherData(rel_file_path, file_base_name, output_file, column_specs, max_part)

    print("\nFinished at", str(datetime.datetime.now()))
