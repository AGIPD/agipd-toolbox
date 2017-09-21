#!/usr/bin/python3

# this script starts mutiple processes in the background on one node

import os
import sys
import argparse
import multiprocessing
from datetime import date
import glob

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_PATH", BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from utils import  create_dir

from dark_and_xray.gather import Gather
from dark_and_xray.xfel_process_dark import ProcessDark
from correct import Correct


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Input dir to read the results from")
    parser.add_argument("--dark_dir",
                        type=str,
                        help="Dir where the dark constants can be found")
    parser.add_argument("--gain_dir",
                        type=str,
                        help="Dir where the gain constants can be found")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Output dir to write the results into")
    parser.add_argument("--run_list",
                        type=str,
                        nargs="+",
                        required=True,
                        help="Run numbers to extract offset data from (has to "
                             "be in the order high, medium, low)")
    parser.add_argument("--type",
                        type=str,
                        required=True,
                        choices=["dark", "correct", "ff-correct"],
                        help="Which type to run: \n"
                             "dark: generating the dark constants\n"
                             "correct: apply constants on experiment data\n`"
                             "ff-correct: apply flat field correction on experiment data")
    parser.add_argument("--energy",
                        type=int,
                        help="Information on photon energy")

    parser.add_argument("--base_cal_dir",
                        type = str,
                        help = "Dir where flatfield calibration constants (agipd_base_cal.h5) can be found")

    args = parser.parse_args()

    if args.type == "dark" and (len(args.run_list) != 3):
        print("Runs for all 3 gain stages are required to calculate dark constants. Quitting.")
        sys.exit(1)

    #if args.type == "correct" and (args.dark_dir is None or args.gain_dir or args.energy is None):
    if args.type == "correct" and (args.dark_dir is None):
        print("Missing dark_dir. Quitting.")
        sys.exit(1)

    if args.type == "ff-correct":
        if args.dark_dir is None:
            print("Missing dark_dir. Quitting.")
            sys.exit(1)
        elif args.base_cal_dir is None:
            print("Missing base_cal_dir. Quitting.")
            sys.exit()
    return args


class StartAnalyse():
    def __init__(self, ana_type, run_list, input_dir, dark_dir, gain_dir, base_cal_fname, output_dir, energy, use_xfel_format):
        self.ana_type = ana_type
        self.run_list = run_list
        self.input_dir = input_dir
        self.dark_dir = dark_dir
        self.gain_dir = gain_dir
        self.base_cal_fname = base_cal_fname
        self.output_dir = output_dir
        self.energy = energy

        #self.number_of_runs = 1
        #self.modules_per_run = 1
        self.number_of_runs = 2
        self.modules_per_run = 16//self.number_of_runs
        self.today = str(date.today())

        self.use_xfel_format = use_xfel_format

        if ana_type == "dark":
            self.output_dir_gather = os.path.join(self.output_dir,
                                                  "gather")
            self.output_dir_process = self.output_dir

            self.run_gather()
            self.run_process(use_xfel_format=True)
            self.run_process(use_xfel_format=False)
        elif self.ana_type == "correct":
            for run_number in self.run_list:
                self.correct(run_number)
        elif self.ana_type == "ff-correct":
            for run_number in self.run_list:
                self.correct(run_number, ctype = "ff")
        

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

#ctypes (correction types):
#agipd: pedestal correction from darks + absolute gain correciton from xray fitting current source dynamic range scan (drscs)
#ff: pedestal correction from darks & current source drs + relative gain correction from flatfield analysis and csdrs

    def correct(self, run_number, ctype = "agipd"):
        print("Correction type: {}".format(ctype))
        for j in range(self.number_of_runs):
            for i in range(self.modules_per_run):
                module = str(j*self.modules_per_run + i).zfill(2)
                print("module", module)

                data_fname_prefix = "RAW-{}-AGIPD{}*".format(run_number.upper(), module)
                data_fname = os.path.join(self.input_dir,
                                          run_number,
                                          data_fname_prefix)

                data_parts = glob.glob(data_fname)
                print("data_parts", data_parts)

                process_list = []
                for data_fname in data_parts:
                    part = int(data_fname[-8:-3])
                    print("part", part)

                    if use_xfel_format:
                        fname_prefix = "dark_AGIPD{}_xfel".format(module)
                    else:
                        fname_prefix = "dark_AGIPD{}_agipd_".format(module)
                    dark_fname_prefix = os.path.join(self.dark_dir, fname_prefix)
                    dark_fname = glob.glob("{}*".format(dark_fname_prefix))
                    if dark_fname:
                        dark_fname = dark_fname[0]
                    else:
                        print("No dark constants found. Quitting.")
                        sys.exit(1)
                    print(dark_fname)

                    if use_xfel_format:
                        fname_prefix = "gain_AGIPD{}_xfel".format(module)
                    else:
                        fname_prefix = "gain_AGIPD{}_agipd_".format(module)

                    gain_fname_prefix = os.path.join(self.gain_dir, fname_prefix)
                    gain_fname = glob.glob("{}*".format(gain_fname_prefix))
                    if ctype == "agipd":
                        if gain_fname:
                            gain_fname = gain_fname[0]
                        else:
                            print("No gain constants found.")
                            #print("No gain constants found. Quitting.")
                            #sys.exit(1)

                    if ctype == "ff":
                        if not glob.glob(self.base_cal_fname):
                            print("No agipd_base_cal.h5 found. Quitting.")
                            sys.exit(1)


                    output_dir = os.path.join(self.output_dir, run_number)
                    create_dir(output_dir)

                    fname = "corrected_AGIPD{}-S{:05d}.h5".format(module, part)
                    output_fname = os.path.join(output_dir, fname)

                    p = multiprocessing.Process(target=Correct, args=(ctype,
                                                                      int(module),
                                                                      data_fname,
                                                                      dark_fname,
                                                                      None,
                                                                      #gain_fname,
                                                                      base_cal_fname,
                                                                      output_fname,
                                                                      self.energy,
                                                                      use_xfel_format))
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

    #input_dir = "/gpfs/exfel/exp/SPB/201701/p002012"
    #run_list = ["r0037", "r0038", "r0039"]
    #output_dir = os.path.join("/gpfs/exfel/exp/SPB/201701/p002012/scratch/user/kuhnm/dark",
    #                          '-'.join(run_list))

    ana_type = args.type
    input_dir = args.input_dir
    dark_dir = args.dark_dir
    #gain_dir = args.gain_dir
    gain_dir = ""
    run_list = args.run_list
    base_cal_fname = None
    if ana_type == "dark":
        output_dir = os.path.join(args.output_dir,
                                  '-'.join(run_list))
    else:
        output_dir = os.path.join(args.output_dir)
    if ana_type == "ff-correct":
        base_cal_fname = os.path.join(args.base_cal_dir,"agipd_base_cal.h5")

    energy = args.energy
    use_xfel_format = True

    print("Running with")
    print("input_dir", input_dir)
    if ana_type == "correct":
        print("dark_dir", dark_dir)
        print("gain_dir", gain_dir)
    if ana_type == "ff-correct":
        print("base_cal_fname", base_cal_fname)
    print("output_dir", output_dir)
    print("run_list", run_list)
    print("type", ana_type)
    print("use_xfel_format for read in", use_xfel_format)

    ana = StartAnalyse(ana_type,
                       run_list,
                       input_dir,
                       dark_dir,
                       gain_dir,
                       base_cal_fname,
                       output_dir,
                       energy,
                       use_xfel_format)
