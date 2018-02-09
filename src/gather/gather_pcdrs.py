import numpy as np
from gather_base import AgipdGatherBase


class AgipdGatherPcdrs(AgipdGatherBase):
    def __init__(self,
                 in_fname,
                 out_fname,
                 runs,
                 preproc_fname=None,
                 max_part=False,
                 asic=None,
                 use_xfel_format=False,
                 backing_store=True):

        self.runs = runs
        self.n_runs = 8

        super().__init__(in_fname=in_fname,
                         out_fname=out_fname,
                         runs=runs,
                         preproc_fname=preproc_fname,
                         max_part=max_part,
                         asic=asic,
                         use_xfel_format=use_xfel_format,
                         backing_store=backing_store)

    def set_pos_indices(self, run_idx):

        # TODO instead of concatenate use two lists

        start = self.n_runs - 1 - run_idx
        stop = self.n_rows // 2
        idx_upper = np.arange(start, stop, self.n_runs)
#        print("idx_upper", idx_upper)

        start = self.n_rows // 2 + run_idx
        stop = self.n_rows
        idx_lower = np.arange(start, stop, self.n_runs)
#        print("idx_lower", idx_lower)

        pos_idx_rows = np.concatenate((idx_upper, idx_lower))
        pos_idx_cols = slice(None)

#        print("pos_idx_rows", pos_idx_rows)
#        print("pos_idx_cols", pos_idx_cols)

        return [[pos_idx_rows, pos_idx_cols]]


if __name__ == "__main__":
    import multiprocessing
    import os
    import sys

    try:
        CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    except:
        CURRENT_DIR = os.path.dirname(os.path.realpath('__file__'))

    BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_DIR))
    SRC_PATH = os.path.join(BASE_PATH, "src")

    if SRC_PATH not in sys.path:
        sys.path.insert(0, SRC_PATH)

    import utils  # noqa E402

    use_xfel_format = True
#    use_xfel_format = False

    if use_xfel_format:
        base_dir = "/gpfs/exfel/exp/SPB/201730/p900009"
#        run_list = [["0488"]]
#        run_list = [[488, 489]]
        run_list = [[488, 489, 490, 491, 492, 493, 494, 495]]
        asic = None

        subdir = "scratch/user/kuhnm"

        number_of_runs = 1
        channels_per_run = 1
        for runs in run_list:
            process_list = []
            for j in range(number_of_runs):
                for i in range(channels_per_run):
                    channel = j * channels_per_run + i
                    in_file_name = ("RAW-R{run_number:04d}-" +
                                    "AGIPD{:02d}".format(channel) +
                                    "-S{part:05d}.h5")
                    in_fname = os.path.join(base_dir,
                                            "raw",
                                            "r{run_number:04d}",
                                            in_file_name)

                    run_subdir = "r" + "-r".join(str(r).zfill(4) for r in runs)
                    out_dir = os.path.join(base_dir,
                                           subdir,
                                           run_subdir,
                                           "gather")
                    utils.create_dir(out_dir)

                    out_file_name = ("{}-AGIPD{}-gathered.h5"
                                     .format(run_subdir.upper(), channel))
                    out_fname = os.path.join(out_dir,
                                             out_file_name)

                    p = multiprocessing.Process(target=AgipdGatherPcdrs,
                                                args=(in_fname,
                                                      out_fname,
                                                      runs,
                                                      False,  # max_part
                                                      asic,
                                                      use_xfel_format))
                    p.start()
                    process_list.append(p)

                for p in process_list:
                    p.join()
