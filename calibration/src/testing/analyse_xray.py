# helper script to develop processing for xray

import os
import sys
from timeit import default_timer as timer


BASE_PATH = "/home/jsibille/agipd/calibration/calibration"

SRC_PATH = os.path.join(BASE_PATH, "src")
GATHER_PATH = os.path.join(SRC_PATH, "gather")
PROCESS_PATH = os.path.join(SRC_PATH, "process")

if GATHER_PATH not in sys.path:
    sys.path.insert(0, GATHER_PATH)

if PROCESS_PATH not in sys.path:
    sys.path.insert(0, PROCESS_PATH)

from gather_base import GatherBase as Gather  # noqa E402
from process_xray import ProcessXray as Process  # noqa E402


def create_dir(directory_name):
    """Creates a directory including supdirectories if it does not exist.

    Args:
        direcoty_name: The path of the direcory to be created.
    """

    if not os.path.exists(directory_name):
        try:
            os.makedirs(directory_name)
            print("Dir '{}' does not exist. Create it."
                  .format(directory_name))
        except IOError:
            if os.path.isdir(directory_name):
                pass


if __name__ == "__main__":
    start = timer()
    module = "M304"
    element = "Mo"
    memcell = 175

    in_base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/raw/333-325-331-304-320-312-302-311"
    subdir = "temperature_m20C/xray"
#    in_base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/raw/315-304-309-314-316-306-307"  # noqa E501
#    subdir = "temperature_m25C/xray"

    in_filename = ("{}*_xray_{}_{}".format(module, element, memcell)
                   + "_{run_number:05}.nxs")
    in_fname = os.path.join(in_base_dir, subdir, in_filename)

    runs = [0]
#    runs = [4]

    out_base_dir = ("/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/jsibille/"
                    "tmp/cfel")
    out_dir = os.path.join(out_base_dir, module, subdir)

    g_out_dir = os.path.join(out_dir, "gather")
    create_dir(g_out_dir)

    g_out_filename = "{}_xray_{}_{}_gathered.h5".format(module, element, memcell)
    g_out_fname = os.path.join(g_out_dir, g_out_filename)

    properties = {
        "measurement": "xray",
        "n_rows_total": 128,
        "n_cols_total": 512,
        "max_pulses": 2,
        "n_memcells": 1
    }

    print("Used parameters:")
    print("in_fname=", in_fname)
    print("out_fname=", g_out_fname)
    print("runs=", runs)
    print("properties", properties)
    print()

    obj = Gather(in_fname=in_fname,
                 out_fname=g_out_fname,
                 runs=runs,
                 properties=properties,
                 use_interleaved=True,
                 preproc_fname=None,
                 max_part=None,
                 asic=None,
                 layout="cfel_layout")

    #obj.run()

    p_out_dir = os.path.join(out_dir, "process")
    create_dir(p_out_dir)

    p_out_filename = "{}_xray_{}_processed.h5".format(module, memcell)
#    p_out_filename = "{}_xray_{}.h5".format(module, element)
    p_out_fname = os.path.join(p_out_dir, p_out_filename)

    print("Used parameters for process:")
    print("in_fname=", g_out_fname)
    print("out_fname", p_out_fname)
    print("runs", runs)

    Process(in_fname=g_out_fname,
            out_fname=p_out_fname,
            runs=runs)


    end = timer()
    print("Total time: ", end - start)
