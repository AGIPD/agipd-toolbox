#!/usr/bin/python3

import os
import h5py
import argparse

import utils

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir",
                        type=str,
                        required=True,
                        help="Base directory to work in")
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Template for input files")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Name of the output file")

    args = parser.parse_args()

    return args


class JoinConstants():
    def __init__(self, in_fname, out_fname):
        self.in_fname = in_fname
        self.out_fname = out_fname

        self.n_channels = 16

        self.source_content = None

        self.run()

    def run(self):

        f = None
        try:
            f = h5py.File(out_fname, "w")

            # TODO change to automatic channel detection
            for channel in range(self.n_channels):
                fname = self.in_fname.format(channel)

                print("loading content of file {}".format(fname))
                file_content = utils.load_file_content(fname)

                prefix = "channel{:02d}".format(channel)
                for key in file_content:
                    f.create_dataset(prefix + "/" + key,
                                     data=file_content[key])

                f.flush()
        finally:
            if f is not None:
                f.close()

if __name__ == "__main__":
    args = get_arguments()

#    base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/dark"
#    in_file_name = "dark_AGIPD{:02d}_agipd_2017-10-27.h5"

    base_dir = args.base_dir
    in_file_name = args.input_file
    in_fname = os.path.join(base_dir, in_file_name)

#    out_file_name = "dark_joined_constants_agipd.h5"
    out_file_name = args.output_file
    out_fname = os.path.join(base_dir, out_file_name)

    JoinConstants(in_fname, out_fname)
