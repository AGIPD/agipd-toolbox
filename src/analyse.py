"""
Attempt at making drscs.py more generic

Called from job_scripts/analyse.sh
   - there all the input arguments are defined

For gather:
- calls GatherData from gather.py

TODO: adjust processing to be generic
For processing:
- calls ParallelProcess from parallel_process.py

"""

from __future__ import print_function

import os
import sys
import argparse
import datetime
import time
import numpy as np
from string import Template

from gather import GatherData
from parallel_process import ParallelProcess
from merge_drscs import ParallelMerge

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
                        choices=range(1, 17),
                        help="Asic number")
    parser.add_argument("--asic_list",
                        type=int,
                        nargs="+",
                        help="List of asics")
    parser.add_argument("--run_type",
                        type=str,
                        required=True,
                        choices=["gather", "process", "merge"],
                        help="What type of run should be started")
    parser.add_argument("--measurement",
                        type=str,
                        required=True,
                        choices=["dark", "xray", "clamped_gain", "drscs", "drscs_dark"],
                        help="Which measurement to analyse: dark, xray, clamped_gain, drscs, drscs_dark")
    parser.add_argument("--column_spec",
                        type=int,
                        nargs='+',
                        default=False,
                        help="Which index files to use for which column, e.g. 9, 10, 11, 12")
    parser.add_argument("--max_part",
                        type=int,
                        default=False,
                        help="Maximum number of parts to be combined")
    parser.add_argument("--reduced_columns",
                        type=int,
                        nargs='+',
                        default=False,
                        help="If only a subset of the columns should be gathered")

    args = parser.parse_args()

    if args.run_type == "gather":
        if args.column_spec and len(args.column_spec) != 4:
            print("There have to be 4 columns defined")
            sys.exit(1)

        if not args.asic:
            print("Asic has to be set for run_type {}".format(args.run_type))
            sys.exit(1)
    elif args.run_type == "process":
        if not args.asic:
            print("Asic has to be set for run_type {}".format(args.run_type))
            sys.exit(1)
    elif args.run_type == "process":
        if not args.asic_list:
            print("Asic list has to be set for run_type {}".format(args.run_type))
            sys.exit(1)

    if args.run_type == "merge" and args.measurement != "drscs":
        print("Merge is only supported for drscs")
        sys.exit(1)

    if args.measurement == "dark" and not args.tint:
        print("The tint must be defined for dark!")
        sys.exit(1)

    if args.measurement == "xray" and not args.element:
        print("The element must be defined for xray!")
        sys.exit(1)

    if args.measurement == "drscs" and not args.current and args.run_type != "merge":
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

class Analyse():
    def __init__(self, run_type, meas_type, input_base_dir, output_base_dir,
                 n_processes, module, temperature, current, tint, element,
                 asic, asic_list, safety_factor, column_spec, reduced_columns, max_part):
        print("started Analyse")

        self.run_type = run_type
        self.meas_type = meas_type
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir
        self.n_processes = n_processes
        self.module = module
        self.temperature = temperature
        self.current = current
        self.tint = tint
        self.element = element
        self.asic = asic
        self.asic_list = asic_list
        self.safety_factor = safety_factor
        self.reduced_columns = reduced_columns

        if column_spec and len(column_spec) == 4:
            # [[<column>, <file index>],...]
            # e.g. for a file name of the form M234_m8_drscs_itestc150_col15_00001_part00000.nxs
            # the entry would be                                         [15,   1]
            #column_specs = [[15, 9], [26, 10], [37, 11], [48, 12]]
            self.column_specs = [[15, column_spec[0]],
                                 [26, column_spec[1]],
                                 [37, column_spec[2]],
                                 [48, column_spec[3]]]
        else:
            self.column_specs = [15, 26, 37, 48]

        if self.reduced_columns:
            self.column_specs = self.reduced_columns

        # the columns for drscs dark are a permutation of the columns of drscs
        # columns 1, 5 injected -> 3, 7 dark
        # columns 2, 6 injected -> 4, 8 dark
        # columns 3, 7 injected -> 1, 5 dark
        # columns 4, 8 injected -> 2, 6 dark
        if self.meas_type == "drscs_dark":
            c = self.column_specs
            self.column_specs = [c[2], c[3], c[0], c[1]]

        self.max_part = max_part

        print("Configured parameter for type {}: ".format(self.run_type))
        print("module: ", self.module)
        print("temperature: ", self.temperature)
        print("measurement: ", self.meas_type)
        print("current (drscs): ", self.current)
        print("tint (dark): ", self.tint)
        print("element (xray): ", self.element)
        print("asic: ", self.asic)
        print("asic_list: ", self.asic_list)
        print("input_dir: ", self.input_base_dir)
        print("output_dir: ", self.output_base_dir)
        print("column_specs: ", self.column_specs)
        print("max_part: ", self.max_part)

        # Usually the input directory and file names correspond to the meas_type
        self.meas_input = {}
        self.meas_input[meas_type] = meas_type
        # but there are exceptions
        self.meas_input["drscs_dark"] = "drscs"

        self.meas_spec = {
            "dark" : self.tint,
            "xray" : self.element,
            "drscs" : self.current,
            "drscs_dark" : self.current,
        }

        self.run()

    def run(self):
        print("\nStarted at", str(datetime.datetime.now()))
        t = time.time()

        if self.run_type == "gather":
            self.run_gather()
        elif self.run_type == "process":
            self.run_process()
        elif self.run_type == "merge":
            self.run_merge_drscs()
        else:
            print("Unsupported argument: run_type {}".format(self.run_type))

        print("\nFinished at", str(datetime.datetime.now()))
        print("took time: ", time.time() - t)


    def run_gather(self):
        module_split = self.module.split("_")

        #TODO: make this work for clamped gain! (in directory: clamped_gain, in filename: cg)
        input_file_name = "{}*_{}_{}".format(module_split[0],
                                             self.meas_input[self.meas_type],
                                             self.meas_spec[self.meas_type])
        input_file_dir = os.path.join(self.input_base_dir,
                                      self.temperature,
                                      self.meas_input[self.meas_type],
                                      self.meas_spec[self.meas_type])
        # TODO: Manu and Jenny have to discuss the dir structure
        # (i.e. if xray and dark should have subdir as in dict)
        # drscs is additionally separated into current directories
        #if self.meas_type == "drscs":
        #    input_file_dir = os.path.join(input_file_dir, self.current)

        input_fname = os.path.join(input_file_dir, input_file_name)

        output_file_name = "{}_{}_{}_asic{}.h5".format(module_split[0],
                                                       self.meas_type,
                                                       self.meas_spec[self.meas_type],
                                                       str(self.asic).zfill(2))
        output_file_dir = os.path.join(self.output_base_dir,
                                       module_split[0],
                                       self.temperature,
                                       self.meas_type,
                                       self.meas_spec[self.meas_type],
                                       self.run_type)
        #TODO: see input_file_dir
        #if meas_type == "drscs": #same as above
        #    output_file_dir = os.path.join(output_file_dir, self.current)

        output_fname = os.path.join(output_file_dir, output_file_name)

        create_dir(output_file_dir)

       # is this necessary? or is there a better way to do this?
        if self.meas_type.startswith("drscs"):
            GatherData(self.asic, input_fname, output_fname, self.meas_type,
                       self.max_part, self.column_specs)
        else:
            GatherData(self.asic, input_fname, output_fname, self.meas_type,
                       self.max_part)

    def run_process(self):
        # the input files for processing are the output ones from gather
        input_file_name = "{}_{}_{}_asic{}.h5".format(self.module,
                                                      self.meas_type,
                                                      self.meas_spec[self.meas_type],
                                                      str(self.asic).zfill(2))
        input_file_dir = os.path.join(self.input_base_dir,
                                      self.module,
                                      self.temperature,
                                      self.meas_type,
                                      self.meas_spec[self.meas_type],
                                      "gather")
        #TODO see gather input_file_dir
        #if meas_type == "drscs":
        #    input_file_dir = os.path.join(input_file_dir, self.current)

        input_fname = os.path.join(input_file_dir, input_file_name)

        output_file_name = ("{}_{}_{}_asic{}_processed.h5"
                            .format(self.module,
                                    self.meas_type,
                                    self.meas_spec[self.meas_type],
                                    str(self.asic).zfill(2)))
        print("output_file_name", output_file_name)

        output_file_dir = os.path.join(self.output_base_dir,
                                       self.module,
                                       self.temperature,
                                       self.meas_type,
                                       self.meas_spec[self.meas_type],
                                       self.run_type)
        #TODO see gather input_file_dir
        #if meas_type == "drscs":
        #    output_file_dir = os.path.join(output_file_dir, self.current)
        output_fname = os.path.join(output_file_dir, output_file_name)

        create_dir(output_file_dir)

        plot_prefix = "{}_{}_asic".format(self.module, self.meas_spec[self.meas_type])
        plot_dir = os.path.join(self.output_base_dir,
                                self.module,
                                self.temperature,
                                self.meas_type,
                                "plots",
                                self.meas_spec[self.meas_type])
        #TODO see gather input_file_dir
        #if meas_type == "drscs":
        #    plot_dir = os.path.join(plot_dir, self.current)
        create_dir(plot_dir)

        pixel_v_list = np.arange(64)
        pixel_u_list = np.arange(64)
        mem_cell_list = np.arange(352)

        proc = ParallelProcess(self.asic,
                               input_fname,
                               pixel_v_list,
                               pixel_u_list,
                               mem_cell_list,
                               self.n_processes,
                               self.safety_factor,
                               output_fname)

    def run_merge_drscs(self):

        base_path = self.input_base_dir
        asic_list = self.asic_list

        input_path = os.path.join(base_path, self.module, self.temperature, "drscs")
        # substitute all except current and asic
        input_template = (Template("${p}/${c}/process/${m}_drscs_${c}_asic${a}_processed.h5")
                          .safe_substitute(p=input_path, m=self.module))
        # make a template out of this string to let Combine set current and asic
        input_template = Template(input_template)

        output_dir = os.path.join(base_path, self.module, self.temperature,
                                   "drscs", "combined")
        output_template = (Template("${p}/${m}_drscs_asic${a}_combined.h5")
                           .safe_substitute(p=output_dir,
                                            m=self.module,
                                            t=self.temperature))
        output_template = Template(output_template)

        create_dir(output_dir)

        ParallelMerge(input_template, output_template, asic_list, self.n_processes)

if __name__ == "__main__":

    args = get_arguments()

    run_type = args.run_type
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
    asic_list = args.asic_list
    column_spec = args.column_spec
    reduced_columns = args.reduced_columns
    max_part = args.max_part
    safety_factor = 1000

    Analyse(run_type, meas_type, input_base_dir, output_base_dir, n_processes,
            module, temperature, current, tint, element, asic, asic_list,
            safety_factor, column_spec, reduced_columns, max_part)
