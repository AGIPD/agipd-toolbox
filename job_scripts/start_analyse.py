import os
import sys
import argparse
import multiprocessing
from analuse import Analyse

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print("BASE_PATH", BASE_PATH)
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)


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
    parser.add_argument("--meas_spec",
                        type=str,
                        help="Measurement specifics used:\n"
                             "drscs: Current to use, e.g. itestc20\n"
                             "dark: Integration time, e.g. tint150ns\n"
                             "xray: Element used for fluorescence, e.g. Cu")
    parser.add_argument("--asic_list",
                        type=int,
                        nargs="+",
                        help="List of asics")
    parser.add_argument("--safety_factor",
                        required=True,
                        type=int,
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
                        choices=["dark", "xray", "clamped_gain", "pcdrs",
                                 "drscs", "drscs_dark", "correct"],
                        help="Which type to run: \n"
                             "dark: generating the dark constants\n"
                             "xray"
                             "clamped_gain"
                             "pcdrs"
                             "drscs"
                             "drscs_dark"
                             "correct: apply constants on experiment data")
    parser.add_argument("--run_list",
                        type=str,
                        nargs="+",
                        required=True,
                        help="Run numbers to extract offset data from (has to "
                             "be in the order high, medium, low)")

    parser.add_argument("--column_spec",
                        type=int,
                        nargs='+',
                        default=False,
                        help="Which index files to use for which column, "
                             "e.g. 9, 10, 11, 12")
    parser.add_argument("--max_part",
                        type=int,
                        default=False,
                        help="Maximum number of parts to be combined")
    parser.add_argument("--reduced_columns",
                        type=int,
                        nargs='+',
                        default=False,
                        help="If only a subset of the columns should be "
                             "gathered")
    parser.add_argument("--current_list",
                        type=str,
                        nargs='+',
                        help="Lists of currents to analyse")
    parser.add_argument("--energy",
                        type=int,
                        help="Information on photon energy")

    args = parser.parse_args()

    if args.run_type == "gather":
        if args.column_spec and len(args.column_spec) != 4:
            print("There have to be 4 columns defined")
            sys.exit(1)

        if not args.asic:
            print("Asic has to be set for run_type {}".format(args.run_type))
            sys.exit(1)
    elif args.run_type == "process":
        if not args.asic_list:
            print("Asic list has to be set for run_type {}"
                  .format(args.run_type))
            sys.exit(1)

    if args.run_type == "merge" and args.type != "drscs":
        print("Merge is only supported for drscs")
        sys.exit(1)

    if ((args.type == "dark" and args.type == "xray" and args.type == "drscs")
            and not args.meas_spec):
        print("The meas_spec must be defined!")
        sys.exit(1)

    if args.type == "dark" and (len(args.run_list) != 3):
        print("Runs for all 3 gain stages are required to calculate dark "
              "constants. Quitting.")
        sys.exit(1)

    return args


class StartAnalyse():
    def __init__(self):
        args = get_arguments()

        self.run_type = args.run_type
        self.meas_type = args.type
        self.input_base_dir = args.input_dir
        self.output_base_dir = args.output_dir
        self.run_list = args.run_list
        self.n_processes = args.n_processes
        self.module = args.module
        self.temperature = args.temperature
        self.meas_spec = args.meas_spec
        self.asic_list = args.asic_list
        self.safety_factor = args.safety_factor
        self.energy = args.energy
        self.max_part = args.max_part
        self.column_spec = args.column_spec
        self.reduced_columns = args.reduced_columns
        self.current_list = args.current_list
        self.energy = args.energy

        self.run()

    def run(self):

        # TODO check for required parameters and stop if they are not set

        jobs = []
        if self.run_type == "merge":
            Analyse(self.run_type,
                    self.meas_type,
                    self.input_base_dir,
                    self.output_base_dir,
                    self.n_processes,
                    self.module,
                    self.temperature,
                    self.meas_spec,
                    0,
                    self.asic_list,
                    self.safety_factor,
                    self.column_spec,
                    self.reduced_columns,
                    self.max_part,
                    self.current_list)
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
                                                  self.meas_spec,
                                                  asic,
                                                  self.asic_list,
                                                  self.safety_factor,
                                                  self.column_spec,
                                                  self.reduced_columns,
                                                  self.max_part))
                jobs.append(p)
                p.start()

            for job in jobs:
                job.join()


if __name__ == "__main__":
    StartAnalyse()
