import os
import numpy as np
from plotting import GeneratePlots
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir",
                        type=str,
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/processed/",
                        help="Processing directory base")
    parser.add_argument("--n_processes",
                        type=int,
                        default=10,
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
    parser.add_argument("--asic",
                        type=int,
                        required=True,
                        choices=range(1, 17),
                        help="Asic number")
    parser.add_argument("--plot_dir",
                        type=str,
                        help="Subdir in which the plots should be stored")
    parser.add_argument("--pixel",
                        type=int,
                        nargs = 3,
                        help="Pixel and memory cell to create a plot from")

    args = parser.parse_args()

    return args


def condition_function(error_code):
    indices = np.where(error_code != 0)

    return indices


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

    base_dir = args.base_dir
    asic = args.asic
    module = args.module
    temperature = args.temperature
    current = args.current
    if args.pixel is not None:
        idx = tuple(args.pixel)
    else:
        idx = None
    print("idx", idx)

    plot_subdir = args.plot_dir or "asic{}_failed".format(str(asic).zfill(2))

    #base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"
    #asic = 3
    #module = "M303"
    #temperature = "temperature_m15C"
    #current = "itestc20"

    #plot_subdir = "asic{}_failed".format(str(asic).zfill(2))
    #plot_subdir = "manu_test"

    n_processes = 10

    gather_fname = os.path.join(base_dir, module, temperature, "drscs", current, "gather",
                                "{}_drscs_{}_asic{}.h5".format(module, current,
                                                                  str(asic).zfill(2)))
    process_fname = os.path.join(base_dir, module, temperature, "drscs", current, "process",
                                  "{}_drscs_{}_asic{}_processed.h5".format(module, current,
                                                                           str(asic).zfill(2)))
    plot_dir = os.path.join(base_dir, module, temperature, "drscs", "plots", current, plot_subdir)
    plot_prefix = "{}_{}".format(module, current)

    create_dir(plot_dir)

    obj = GeneratePlots(asic, gather_fname, plot_prefix, plot_dir, n_processes)

    #idx = (5,5,1)

    if idx is not None:
        obj.run_idx(idx)
    else:
        obj.run_condition(process_fname, condition_function)
