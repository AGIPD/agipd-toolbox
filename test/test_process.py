import sys
import os

try:
    CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
except:
    CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

BASE_PATH = os.path.dirname(CURRENT_DIR)
SRC_PATH = os.path.join(BASE_PATH, "src")
PROCESS_PATH = os.path.join(SRC_PATH, "process")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

if PROCESS_PATH not in sys.path:
    sys.path.insert(0, PROCESS_PATH)

import utils

from process_dark import AgipdProcessDark
from process_pcdrs import AgipdProcessPcdrs

def call_xfel_mode(run_type,
                   in_base_dir,
                   out_base_dir,
                   run_list,
                   channel,
                   use_xfel_format=True):

    if run_type == "dark":
        Process = AgipdProcessDark
    elif run_type == "pcdrs":
        Process = AgipdProcessPcdrs

    in_file_name = ("R{run_number:04d}-" +
                    "AGIPD{:02d}-gathered.h5".format(channel))
    in_fname = os.path.join(in_base_dir,
                            "r{run_number:04d}",
                            "gather",
                            in_file_name)

    out_dir = os.path.join(out_base_dir, run_type)
    utils.create_dir(out_dir)

    if use_xfel_format:
        fname = "{}_AGIPD{:02d}_xfel.h5".format(run_type, channel)
    else:
        fname = "{}_AGIPD{:02d}_agipd.h5".format(run_type, channel)

    out_fname = os.path.join(out_dir, fname)

    print("Used parameter for {} run:".format(run_type))
    print("in_fname=", in_fname)
    print("out_fname", out_fname)
    print("runs", run_list)
    print("use_xfel_format=", use_xfel_format)
    print()

    obj = Process(in_fname=in_fname,
                  out_fname=out_fname,
                  runs=run_list,
                  use_xfel_format=use_xfel_format)


def call_cfel_mode(run_type,
                   in_base_dir,
                   out_base_dir,
                   module,
                   temperature,
                   meas_spec,
                   run_names,
                   use_xfel_format=False):

    if run_type == "dark":
        Process = AgipdProcessDark
    elif run_type == "pcdrs":
        Process = AgipdProcessPcdrs

    in_file_name = ("{}_{}_".format(module, run_type) +
                    "{run_number}_gathered.h5")
    in_fname = os.path.join(in_base_dir,
                            module,
                            temperature,
                            run_type,
                            meas_spec,
                            "gather",
                            in_file_name)

    out_dir = os.path.join(out_base_dir,
                           module,
                           temperature,
                           run_type,
                           meas_spec)
    utils.create_dir(out_dir)

    fname = "{}_{}_agipd.h5".format(run_type, module)
    out_fname = os.path.join(out_dir, fname)

    print("Used parameter for {} run:".format(run_type))
    print("in_fname", in_fname)
    print("out_fname", out_fname)
    print("runs", run_names)
    print("use_xfel_format", use_xfel_format)
    print()

    obj = Process(in_fname=in_fname,
                  out_fname=out_fname,
                  runs=run_names,
                  use_xfel_format=use_xfel_format)


# TESTS
def test_dark(use_xfel_format):
    run_type = "dark"

    if use_xfel_format:
    #    use_xfel_output_format = False
        use_xfel_output_format = True

        in_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
        out_base_dir = in_base_dir
        run_list = [428, 429, 430]

        channel = 0
        print("channel", channel)

        call_xfel_mode(run_type=run_type,
                       in_base_dir=in_base_dir,
                       out_base_dir=in_base_dir,
                       run_list=run_list,
                       channel=channel,
                       use_xfel_format=use_xfel_output_format)
    else:
        in_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/tmp/cfel"
        out_base_dir = in_base_dir

        module = "M304"
        temperature = "temperature_m25C"
        nature = "tint150ns"

        run_names = ["high", "med", "low"]
#        run_names = ["high"]

        call_cfel_mode(run_type=run_type,
                       in_base_dir=in_base_dir,
                       out_base_dir=in_base_dir,
                       module=module,
                       temperature=temperature,
                       meas_spec=nature,
                       run_names=run_names)

def test_pcdrs():
    in_base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm"
    out_base_dir = in_base_dir
    run_list = ["r0488-r0489-r0490-r0491-r0492-r0493-r0494-r0495"]
    run_type = "pcdrs"

    use_xfel_format = False
#    use_xfel_format = True

    channel = 1
    print("channel", channel)

    call_xfel_mode(run_type=run_type,
                   in_base_dir=in_base_dir,
                   out_base_dir=in_base_dir,
                   run_list=run_list,
                   channel=channel,
                   use_xfel_format=use_xfel_format)

if __name__ == "__main__":
    test_dark(use_xfel_format=False)
    #test_dark(use_xfel_format=True)
    #test_pcdrs()
