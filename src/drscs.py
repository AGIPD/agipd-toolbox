from __future__ import print_function

import os
import sys
from process_data_per_asic import ProcessDrscs
from gather_data_per_asic import GatherData
import argparse
import datetime
import numpy as np

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
    parser.add_argument("--type",
                        type=str,
                        required=True,
                        choices=["gather", "process"],
                        help="What type of run should be started")
    parser.add_argument("--column_spec",
                        type=int,
                        nargs='+',
                        help="Which index files to use for which column, e.g. 9, 10, 11, 12")
    parser.add_argument("--max_part",
                        type=int,
                        default=False,
                        help="Maximum number of parts to be combined")

    args = parser.parse_args()

    if args.type == "gather":
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


if __name__ == "__main__":

    args = get_arguments()

    run_type = args.type
    input_base_dir = args.input_dir
    output_base_dir = args.output_dir
    module = args.module
    temperature = args.temperature
    current = args.current
    asic = args.asic

    #module = "M314"
    #temperature = "temperature_m15C"
    #current = "itestc150"
    #asic = 1
    #input_base_dir = "/gpfs/cfel/fsds/labs/processed/calibration/processed/"

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

    print("Configured parameter for type {}: ".format(run_type))
    print("module: ", module)
    print("temperature: ", temperature)
    print("current: ", current)
    print("asic: ", asic)
    print("input_dir: ", input_base_dir)
    print("output_dir: ", output_base_dir)
    print("column_specs: ", column_specs)
    print("max_part: ", max_part)

    if run_type == "gather":
        module_split = module.split("_")

        input_file_name = "{}*_drscs_{}".format(module_split[0], current)
        input_file_dir = os.path.join(input_base_dir, temperature, "drscs", current)
        input_file = os.path.join(input_file_dir, input_file_name)

        output_file_name = "{}_drscs_{}_asic{}.h5".format(module_split[0], current, str(asic).zfill(2))
        output_file_dir = os.path.join(output_base_dir, module_split[0], temperature, "drscs", current)
        output_file = os.path.join(output_file_dir, output_file_name)

        create_dir(output_file_dir)

        print("\nStarted at", str(datetime.datetime.now()))

        GatherData(asic, input_file, output_file, column_specs, max_part)
        #GatherData(rel_file_path, input_file_name, output_file, column_specs, max_part)

    else:
        # the input files for processing are the output ones from gather
        input_file_name = "{}_drscs_{}_asic{}.h5".format(module, current, str(asic).zfill(2))
        input_file_dir = os.path.join(input_base_dir, module, temperature, "drscs", current)
        input_file = os.path.join(input_file_dir, input_file_name)

        output_file_name = "{}_drscs_{}_asic{}_processed.h5".format(module, current, str(asic).zfill(2))
        output_file_dir = os.path.join(output_base_dir, module, temperature, "drscs", current)
        output_file = os.path.join(output_file_dir, output_file_name)

        plot_dir = os.path.join(output_file_dir, "plots")
        create_dir(plot_dir)

        create_error_plots = False
        pixel_v_list = np.arange(64)
        pixel_u_list = np.arange(64)
        mem_cell_list = np.arange(352)

        print("\nStarted at", str(datetime.datetime.now()))

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
