#!/usr/bin/python3

import os
import h5py
import argparse
import glob

import utils

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="Input directory to load data from")
    parser.add_argument("--input_file",
                        type=str,
                        required=True,
                        help="Template for input files")
    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="Output directory to write results to")
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

        self.source_content = None

    def get_file_names(self):
        fname = self.in_fname.replace("{:02}", "*")

        file_list = glob.glob(fname)
        file_list.sort()

        return file_list

    def run(self):
        file_list = self.get_file_names()

        f = None
        try:
            f = h5py.File(self.out_fname, "w")

            # TODO change to automatic channel detection
            for channel, fname in enumerate(file_list):

                print("channel{}: loading content of file {}".format(channel, fname))
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

    in_dir = args.input_dir
    in_file_name = args.input_file
    in_fname = os.path.join(in_dir, in_file_name)

#    out_file_name = "dark_joined_constants_agipd.h5"
    out_file_name = args.output_file
    out_dir = args.output_dir
    out_fname = os.path.join(out_dir, out_file_name)

    obj = JoinConstants(in_fname, out_fname)
    obj.run()
