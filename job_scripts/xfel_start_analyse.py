#!/usr/bin/python3

# this script starts mutiple processes in the background on one node

import os
import sys
import argparse
import multiprocessing
from datetime import date

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_PATH", BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils import  create_dir

from dark_and_xray.gather import Gather
from dark_and_xray.xfel_process_dark import ProcessDark


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_base_dir",
                        type=str,
                        required=True,
                        help="Input dir to read the results from")
    parser.add_argument("--output_base_dir",
                        type=str,
                        required=True,
                        help="Output dir to write the results into")
    parser.add_argument("--run_list",
                        type=str,
                        nargs=3,
                        required=True,
                        help="Run numbers to extract offset data from (has to "
                             "be in the order high, medium, low)")

    args = parser.parse_args()

    return args


class StartAnalyse():
    def __init__(self, run_list, input_base_dir, output_base_dir, use_xfel_format):
        self.run_list = run_list
        self.input_base_dir = input_base_dir
        self.output_base_dir = output_base_dir

        self.number_of_runs = 2
        self.modules_per_run = 16//self.number_of_runs
        self.today = str(date.today())

        self.input_dir = os.path.join(self.input_base_dir,
                                      "raw")
        self.output_dir_gather = os.path.join(self.output_base_dir,
                                              "gather")
        self.output_dir_process = self.output_base_dir

        self.use_xfel_format = use_xfel_format

    def run_gather(self):
        for run_number in self.run_list:
            process_list = []
            for j in range(self.number_of_runs):
                for i in range(self.modules_per_run):
                    module = str(j*self.modules_per_run + i).zfill(2)

                    fname = "RAW-{}-AGIPD{}-".format(run_number.upper(), module) + "S{:05d}.h5"
                    input_fname = os.path.join(self.input_dir, run_number, fname)

                    create_dir(self.output_dir_gather)
                    output_fname = os.path.join(self.output_dir_gather,
                                                "{}-AGIPD{}-gathered.h5".format(run_number.upper(), module))

                    p = multiprocessing.Process(target=Gather,
                                                args=(input_fname, output_fname, self.use_xfel_format))
                    p.start()
                    process_list.append(p)

                for p in process_list:
                    p.join()

    def run_process(self, use_xfel_format=False):
        self.use_xfel_format = use_xfel_format

        for j in range(self.number_of_runs):
            process_list = []
            for i in range(self.modules_per_run):
                module = str(j*self.modules_per_run + i).zfill(2)
                print("module", module)

                input_fname_list = []
                for run_number in self.run_list:
                    fname = "{}-AGIPD{}-gathered.h5".format(run_number.upper(), module)
                    input_fname = os.path.join(self.output_dir_gather,
                                               fname)
                    input_fname_list.append(input_fname)

                create_dir(self.output_dir_process)

                if self.use_xfel_format:
                    fname = "dark_AGIPD{}_xfel_{}.h5".format(module, self.today)
                else:
                    fname = "dark_AGIPD{}_agipd_{}.h5".format(module, self.today)

                output_fname = os.path.join(self.output_dir_process, fname)

                p = multiprocessing.Process(target=ProcessDark, args=(input_fname_list,
                                                                      output_fname,
                                                                      self.use_xfel_format))
                p.start()
                process_list.append(p)

            for p in process_list:
                p.join()

    def cleanup(self):
        # remove gather dir
        #self.output_dir_gather
        pass


if __name__ == "__main__":
    args = get_arguments()

    #input_base_dir = "/gpfs/exfel/exp/SPB/201701/p002012"
    #run_list = ["r0037", "r0038", "r0039"]
    #output_base_dir = os.path.join("/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm/dark",
    #                               '-'.join(run_list))

    input_base_dir = args.input_base_dir
    run_list = args.run_list
    output_base_dir = os.path.join(args.output_base_dir,
                                   '-'.join(run_list))
    use_xfel_format = True

    print("Running with")
    print("input_base_dir", input_base_dir)
    print("output_base_dir", output_base_dir)
    print("run_list", run_list)
    print("use_xfel_format for read in", use_xfel_format)

    ana = StartAnalyse(run_list, input_base_dir, output_base_dir, use_xfel_format)

#    ana.run_gather()
    ana.run_process(use_xfel_format=True)
    ana.run_process(use_xfel_format=False)
