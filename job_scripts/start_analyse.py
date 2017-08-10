#!/usr/bin/python3

# this script starts mutiple processes in the background on one node

import os
import sys
import argparse
import subprocess
import multiprocessing

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--script_base_dir",
                        type=str,
                        required=True,
                        help="Location of the git repository")
    parser.add_argument("--run_type",
                        type=str,
                        required=True,
                        choices=["gather", "process", "merge"],
                        help="what type of run should be started")
    parser.add_argument("--measurement",
                        type=str,
                        required=True,
                        choices=["dark", "xray", "clamped_gain", "drscs", "drscs_dark"],
                        help="Which measurement to analyse: dark, xray, clamped_gain, drscs, drscs_dark")
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
    parser.add_argument("--asic_list",
                        required=True,
                        type=int,
                        nargs='+',
                        help="Asic number")

    parser.add_argument("--column_spec",
                        type=int,
                        nargs='+',
                        help="Which index files to use for which column, e.g. 9, 10, 11, 12")
    parser.add_argument("--max_part",
                        default=False,
                        type=int,
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

class StartAnalyse():
    def __init__(self):
        args = get_arguments()

        self.base_dir = args.script_base_dir
        self.run_type = args.run_type
        self.meas_type = args.measurement
        self.input_base_dir = args.input_dir
        self.output_base_dir = args.output_dir
        self.n_processes = args.n_processes
        self.module = args.module
        self.temperature = args.temperature
        self.current = args.current
        self.tint = args.tint
        self.element = args.element
        self.asic_list = args.asic_list
        self.max_part = args.max_part
        self.column_spec = args.column_spec
        self.reduced_columns = args.reduced_columns

        self.run()

    def run(self):

        #TODO check for required parameters and stop if they are not set

        script_dir = os.path.join(self.base_dir, "src")

        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        from analyse import Analyse

        jobs = []
        if self.run_type == "merge":
            Analyse(self.run_type,
                    self.meas_type,
                    self.input_base_dir,
                    self.output_base_dir,
                    self.n_processes,
                    self.module,
                    self.temperature,
                    self.current,
                    self.tint,
                    self.element,
                    0,
                    self.asic_list,
                    self.column_spec,
                    self.reduced_columns,
                    self.max_part)
        else:
            for asic in self.asic_list:
                print("Starting script for asic {}\n".format(asic))

                p = multiprocessing.Process(target=Analyse,
                                            args=(self.run_type,
                                                  self.meas_type,
                                                  self.input_base_dir,
                                                  self.output_base_dir,
                                                  self.n_processes,
                                                  self.module,
                                                  self.temperature,
                                                  self.current,
                                                  self.tint,
                                                  self.element,
                                                  asic,
                                                  self.asic_list,
                                                  self.column_spec,
                                                  self.reduced_columns,
                                                  self.max_part))
                jobs.append(p)
                p.start()

            for job in jobs:
                job.join()


if __name__ == "__main__":
    StartAnalyse()
