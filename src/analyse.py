"""
Attempt at making drscs.py more generic

Called from job_scripts/analyse.sh (doesn't exist yet)
   - there all the input arguments are defined

For gather:
- calls GatherData from gather_data_per_asic_generic.py

TODO: adjust processing to be generic
For processing:
- calls ProcessDrscs from process_data_per_asic.py

"""

from __future__ import print_function

import os
import sys
from parallel_process import ParallelProcessing
from gather_data_per_asic import GatherData
import argparse
import datetime
import time
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
    parser.add_argument("--n_processes",
                        type=int,
                        help="The number of processes for the pool")
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
                        help="Current to use, e.g. itestc20")
    parser.add_argument("--tint",
                        type=str,
                        help="Integration time, e.g. tint150ns")
    parser.add_argument("--element",
                        type=str,
                        help="Element used for fluorescence, e.g. Cu")
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
    parser.add_argument("--measurement",
                        type=str,
                        required=True,
                        choices=["dark", "xray", "clamped_gain", "drscs"],
                        help="Which measurement to analyse: dark, xray, clamped_gain, drscs")
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

    if args.measurement == "dark" and not args.tint:
        print("The tint must be defined for dark!")
        sys.exit(1)

    if args.measurement == "xray" and not args.element:
        print("The element must be defined for xray!")
        sys.exit(1)

    if args.measurement == "drscs" and not args.current:
        print("The current must be defined for drscs!")
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
    meas_type = args.measurement
    input_base_dir = args.input_dir
    output_base_dir = args.output_dir
    n_processes = args.n_processes
    module = args.module
    temperature = args.temperature
    current = args.current
    tint = args.tint
    element = args.element
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

    print("Configured parameter for type {}: ".format(run_type))
    print("module: ", module)
    print("temperature: ", temperature)
    print("measurement: ", meas_type)
    print("current (drscs): ", current)
    print("tint (dark): ", tint)
    print("element (xray): ", element)
    print("asic: ", asic)
    print("input_dir: ", input_base_dir)
    print("output_dir: ", output_base_dir)
    print("column_specs: ", column_specs)
    print("max_part: ", max_part)

    meas_spec = {
        "dark" : tint,
        "xray" : element,
        "drscs" : current,
    }

    if run_type == "gather":
        module_split = module.split("_")

        #TODO: make this work for clamped gain! (in directory: clamped_gain, in filename: cg)
        input_file_name = "{}*_{}_{}".format(module_split[0], meas_type, meas_spec[meas_type])
        input_file_dir = os.path.join(input_base_dir,
                                      temperature,
                                      meas_type,
                                      meas_spec[meas_type])
        # TODO: Manu and Jenny have to discuss the dir structure
        # (i.e. if xray and dark should have subdir as in dict)
        # drscs is additionally separated into current directories
        #if meas_type == "drscs":
        #    input_file_dir = os.path.join(input_file_dir, current)

        input_fname = os.path.join(input_file_dir, input_file_name)

        output_file_name = "{}_{}_{}_asic{}.h5".format(module_split[0],
                                                       meas_type,
                                                       meas_spec[meas_type],
                                                       str(asic).zfill(2))
        output_file_dir = os.path.join(output_base_dir,
                                       module_split[0],
                                       temperature,
                                       meas_type,
                                       meas_spec[meas_type],
                                       run_type)
        #TODO: see input_file_dir
        #if meas_type == "drscs": #same as above
        #    output_file_dir = os.path.join(output_file_dir, current)

        output_fname = os.path.join(output_file_dir, output_file_name)

        create_dir(output_file_dir)

        print("\nStarted at", str(datetime.datetime.now()))
        t = time.time()

       # is this necessary? or is there a better way to do this?
        if meas_type == "drscs":
            GatherData(asic, input_fname, output_fname, meas_type, max_part, column_specs)
        else:
            GatherData(asic, input_fname, output_fname, meas_type, max_part)

    else:
        # the input files for processing are the output ones from gather
        input_file_name = "{}_{}_{}_asic{}.h5".format(module,
                                                      meas_type,
                                                      meas_spec[meas_type],
                                                      str(asic).zfill(2))
        input_file_dir = os.path.join(input_base_dir,
                                      module,
                                      temperature,
                                      meas_type,
                                      meas_spec[meas_type],
                                      "gather")
        #TODO see gather input_file_dir
        #if meas_type == "drscs":
        #    input_file_dir = os.path.join(input_file_dir, current)

        input_fname = os.path.join(input_file_dir, input_file_name)

        output_file_name = "{}_{}_{}_asic{}_processed.h5".format(module,
                                                                 meas_type,
                                                                 meas_spec[meas_type],
                                                                 str(asic).zfill(2))
        print("output_file_name", output_file_name)
        output_file_dir = os.path.join(output_base_dir,
                                       module,
                                       temperature,
                                       meas_type,
                                       meas_spec[meas_type],
                                       run_type)
        #TODO see gather input_file_dir
        #if meas_type == "drscs":
        #    output_file_dir = os.path.join(output_file_dir, current)
        output_fname = os.path.join(output_file_dir, output_file_name)

        create_dir(output_file_dir)

        plot_prefix = "{}_{}_asic".format(module, meas_spec[meas_type])
        plot_dir = os.path.join(output_base_dir,
                                module,
                                temperature,
                                meas_type,
                                "plots",
                                meas_spec[meas_type])
        #TODO see gather input_file_dir
        #if meas_type == "drscs":
        #    plot_dir = os.path.join(plot_dir, current)
        create_dir(plot_dir)

        pixel_v_list = np.arange(64)
        pixel_u_list = np.arange(64)
        mem_cell_list = np.arange(352)

        print("\nStarted at", str(datetime.datetime.now()))
        t = time.time()

        proc = ParallelProcessing(asic, input_fname, pixel_v_list, pixel_u_list,
                                  mem_cell_list, n_processes, output_fname)

    print("\nFinished at", str(datetime.datetime.now()))
    print("took time: ", time.time() - t)
