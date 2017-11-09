#!/usr/bin/python3

import os
import sys
import argparse
import multiprocessing

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_PATH", BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")
GATHER_PATH = os.path.join(SRC_PATH, "gather")
PROCESS_PATH = os.path.join(SRC_PATH, "process")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from analyse import Analyse  # noqa E402


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
                        default=1,
                        help="The number of processes for the pool")
    parser.add_argument("--module",
                        type=str,
                        help="Module to gather, e.g M310 (AGIPD lab)")
    parser.add_argument("--channel",
                        type=int,
                        nargs="+",
                        help="Module to gather, e.g 1 (XFEL)")
    parser.add_argument("--temperature",
                        type=str,
                        default="",
                        help="temperature to gather, e.g. temperature_30C"
                             "(only needed for AGIPD lab data)")
    parser.add_argument("--meas_spec",
                        type=str,
                        default=None,
                        help="Measurement specifics used "
                             "(only needed for AGIPD lab data):\n"
                             "dark: Integration time, e.g. tint150ns\n")
#                             "drscs: Current to use, e.g. itestc20\n"
#                             "xray: Element used for fluorescence, e.g. Cu")
    parser.add_argument("--asic_list",
                        type=int,
                        nargs="+",
                        help="List of asics")
    parser.add_argument("--safety_factor",
                        type=int,
                        default=None,
                        help="Safty factor used in the determination of the "
                             "gain stage changes of the analog signal "
                             "(used in drscs)")
    parser.add_argument("--run_type",
                        type=str,
                        required=True,
                        choices=["gather", "process", "merge"],
                        help="What type of run should be started")
    parser.add_argument("--type",
                        type=str,
                        required=True,
                        choices=["dark", "pcdrs"],
#                        choices=["dark", "xray", "clamped_gain", "pcdrs",
#                                 "drscs", "drscs_dark", "correct"],
                        help="Which type to run: \n"
                             "dark: generating the dark constants\n"
                             "pcdrs")
#                             "xray"
#                             "clamped_gain"
#                             "drscs"
#                             "drscs_dark"
#                             "correct: apply constants on experiment data")
    parser.add_argument("--run_list",
                        type=int,
                        nargs="+",
                        required=True,
                        help="Run numbers to extract data from. Requirements: \n"
                             "dark: 3 runs for the gain stages "
                             "high, medium, low (in this order)\n"
                             "pcdrs: 8 runs")

    parser.add_argument("--max_part",
                        type=int,
                        default=None,
                        help="Maximum number of parts to be combined")
    parser.add_argument("--current_list",
                        type=str,
                        nargs='+',
                        help="Lists of currents to analyse (used for drscs)")
    parser.add_argument("--energy",
                        type=int,
                        help="Information on photon energy")
    parser.add_argument("--use_xfel_in_format",
                        action="store_true",
                        default=False,
                        help="Flag describing if the input data is in xfel format")
    parser.add_argument("--use_xfel_out_format",
                        action="store_true",
                        default=False,
                        help="Flag describing if the output data should be "
                             "stored in xfel format")

    args = parser.parse_args()

    if args.module is None and args.channel is None:
        msg = ("one of the two folling arguments is required: "
               "--channel or --module")
        parser.error(msg)
    elif args.module is not None and args.channel is not None:
        msg = ("only one of the two folling arguments is required: "
               "--channel or --module")
        parser.error(msg)

    if args.type == "drscs":
        if len(args.run_list) != 4:
            msg = ("There have to be 4 runs defined "
                   "(each containing 2 columns)")
            parser.error(msg)

#    if args.run_type == "gather":
#        if not args.asic:
#            print("Asic has to be set for run_type {}".format(args.run_type))
#            sys.exit(1)
#    elif args.run_type == "process":
#        if not args.asic_list:
#            print("Asic list has to be set for run_type {}"
#                  .format(args.run_type))
#            sys.exit(1)

#    if args.run_type == "merge" and args.type != "drscs":
#        print("Merge is only supported for drscs")
#        sys.exit(1)

    if (not args.use_xfel_in_format and
            (args.type == "dark" and
                args.type == "xray" and
                args.type == "drscs")
            and not args.meas_spec):
        msg = "The meas_spec must be defined!"
        parser.error(msg)

    if args.type == "dark" and args.type == "gather" and (len(args.run_list) != 1):
        msg = ("Gathering only one run at a time for type dark. Quitting.")
        parser.error(msg)

    if args.type == "dark" and args.type == "process" and (len(args.run_list) != 3):
        msg = ("Runs for all 3 gain stages are required to calculate dark "
               "constants. Quitting.")
        parser.error(msg)

    if args.type == "pcdrs" and (len(args.run_list) != 8):
        msg = ("Pulse capacitor requires 8 runs to calculate constants. "
               "Quitting.")
        parser.error(msg)

    return args


class StartAnalyse():
    def __init__(self):
        args = get_arguments()

        self.run_type = args.run_type
        self.meas_type = args.type
        self.in_base_dir = args.input_dir
        self.out_base_dir = args.output_dir
        self.run_list = args.run_list
        self.n_processes = args.n_processes
        self.module = args.module or args.channel
        self.temperature = args.temperature
        self.meas_spec = args.meas_spec
        self.asic_list = args.asic_list or [None]
        self.safety_factor = args.safety_factor
        self.max_part = args.max_part
        self.current_list = args.current_list if args.current_list else None
        self.energy = args.energy
        self.use_xfel_in_format = args.use_xfel_in_format
        self.use_xfel_out_format = args.use_xfel_out_format

        print("====== Configured parameter in class StartAnalyse ======")
        print("meas_type {}:".format(self.meas_type))
        print("module/channel: ", self.module)
        print("in_dir: ", self.in_base_dir)
        print("out_dir: ", self.out_base_dir)
        print("run_list: ", self.run_list)
        print("n_processes: ", self.n_processes)
        print("temperature: ", self.temperature)
        print("meas_spec: ", self.meas_spec)
        print("asic_list: ", self.asic_list)
        print("safety_factor: ", self.safety_factor)
        print("max_part: ", self.max_part)
        print("current_list: ", self.current_list)
        print("energy: ", self.energy)
        print("use_xfel_in_format: ", self.use_xfel_in_format)
        print("use_xfel_out_format: ", self.use_xfel_out_format)
        print("========================================================")
        self.run()

    def run(self):

        # TODO check for required parameters and stop if they are not set

        jobs = []
        if self.run_type == "merge":
            Analyse(self.run_type,
                    self.meas_type,
                    self.in_base_dir,
                    self.out_base_dir,
                    self.n_processes,
                    self.module,
                    self.temperature,
                    self.meas_spec,
                    0,
                    self.asic_list,
                    self.safety_factor,
                    self.run_list,
                    self.max_part,
                    self.current_list,
                    self.use_xfel_in_format,
                    self.use_xfel_out_format)
        else:
            for m in self.module:
                for asic in self.asic_list:
                    print("Starting script for asic {}\n".format(asic))

                    p = multiprocessing.Process(target=Analyse,
                                                args=(self.run_type,
                                                      self.meas_type,
                                                      self.in_base_dir,
                                                      self.out_base_dir,
                                                      self.n_processes,
                                                      m,
                                                      self.temperature,
                                                      self.meas_spec,
                                                      asic,
                                                      self.asic_list,
                                                      self.safety_factor,
                                                      self.run_list,
                                                      self.max_part,
                                                      self.current_list, # = None
                                                      self.use_xfel_in_format,
                                                      self.use_xfel_out_format))
                    jobs.append(p)
                    p.start()

            for job in jobs:
                job.join()


if __name__ == "__main__":
    StartAnalyse()
