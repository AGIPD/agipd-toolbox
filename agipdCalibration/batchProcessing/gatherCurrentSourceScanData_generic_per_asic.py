from __future__ import print_function

import os
import sys
from gatherdata_per_asic import GatherData
import argparse
import datetime

OUTPUT_BASE_PATH = "/gpfs/cfel/fsds/labs/processed/calibration/processed/"

def get_arguments():
    global OUTPUT_BASE_PATH

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_path",
                        type=str,
                        required=True,
                        help="Relative path under {} where the data can be found".format(OUTPUT_BASE_PATH))
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
                        help="which index files to use for which column, e.g.9, 10, 11, 12")
    parser.add_argument("--asic",
                        type=int,
                        required=True,
                        choices=range(1, 17),
                        help="Asic number")
    parser.add_argument("--max_part",
                        type=int,
                        default=False,
                        help="Maximum number of parts to be combined")

    args = parser.parse_args()

    if args.column_spec and len(args.column_spec) != 4:
        print("There have to be 4 columns defined")
        sys.exit(1)

    return args

def create_dir(directory_name):
    if not os.path.exists(directory_name):
        try:
            os.makedirs(directory_name)
            print("Dir '{0}' does not exist. Create it.".format(directory_name))
        except IOError:
            if os.path.isdir(directory_name):
                pass


def find_module_position():
    pass


if __name__ == "__main__":

    args = get_arguments()

    input_base_path = args.input_path
    module = args.module
    temperature = args.temperature
    current = args.current
    asic = args.asic

    if args.column_spec:
        # [[<column>, <file index>],...]
        # e.g. for a file name of the form M234_m8_drscs_itestc150_col15_00001_part00000.nxs
        # the entry would be                                         [15,   1]
        #column_specs = [[15, 9], [26, 10], [37, 11], [48, 12]]
        column_specs = [[15, args.column_spec[0]],
                        [26, args.column_spec[1]],
                        [37, args.column_spec[2]],
                        [48, args.column_spec[3]]]
    else:
        column_specs = [15, 26, 37, 48]

    max_part = args.max_part
    #False
    #max_part = 10

    print ("configured parameter: ")
    print ("module: ", module)
    print ("temperature: ", temperature)
    print ("current: ", current)
    print ("column_specs: ", column_specs)
    print ("max_part: ", max_part)
    print ("asic: ", asic)
    print ("\nStarted at", str(datetime.datetime.now()))

    #rel_file_path = "311-312-301-300-310-234/temperature_m20C/drscs/itestc20"
    rel_file_path = os.path.join(input_base_path, temperature, "drscs", current)

    module_split = module.split("_")

    file_base_name = "{}*_drscs_{}".format(module_split[0], current)
    #file_base_name = "M301_m3_drscs_itestc150"

    output_file_name = "{}_drscs_{}_asic{}.h5".format(module_split[0], current, asic)
    #output_file_path = "/gpfs/cfel/fsds/labs/processed/calibration/processed/M310/temperature_30C/drscs/itestc20"
    output_file_path = os.path.join(OUTPUT_BASE_PATH, module_split[0], temperature, "drscs", current)
    output_file = os.path.join(output_file_path, output_file_name)

    create_dir(output_file_path)

    GatherData(asic, rel_file_path, file_base_name, output_file, column_specs, max_part)
    #GatherData(rel_file_path, file_base_name, output_file, column_specs, max_part)

    print("\nFinished at", str(datetime.datetime.now()))
