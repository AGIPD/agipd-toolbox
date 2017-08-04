import os
import numpy as np
from plotting import GeneratePlots

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

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"

    asic = 15
    module = "M314"
    temperature = "temperature_m15C"
    current = "itestc80"
    #current = "itestc150"

    #plot_subdir = "asic{}_failed".format(str(asic).zfill(2))
    plot_subdir = "manu_test"

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

    idx = None

    idx = (2,38,99)

    if idx is not None:
        obj.run_idx(idx)
    else:
        obj.run_condition(process_fname, condition_function)
