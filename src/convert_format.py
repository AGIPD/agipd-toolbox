#!/usr/bin/python3

import os
import argparse
import h5py

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
                        help="Name of the input files")
    parser.add_argument("--output_file",
                        type=str,
                        required=True,
                        help="Template of the output file")
    parser.add_argument("--channel",
                        type=int,
                        required=True,
                        help="Module to gather, e.g 1 (XFEL)")
    parser.add_argument("--output_format",
                        type=str,
                        required=True,
                        help="Format to convert the data to (xfel or agipd)")

    args = parser.parse_args()

    return args


class ConvertFormat():
    def __init__(self, input_fname, output_fname, output_format, channel, key_list=None):
        self.input_fname = input_fname
        self.output_fname = output_fname
        self.output_format = output_format
        self.channel = channel

        if key_list is None:
            f = None
            try:
                f = h5py.File(self.input_fname, "r")

                key_list = list(f.keys())
            finally:
                if f is not None:
                    f.close()

            key_list.remove("collection")
            self.keys_to_convert = key_list

        else:
            self.keys_to_convert = key_list
        print("Keys to convert: {}\n".format(self.keys_to_convert))

    def run(self):
        print("Loading input_file from {}".format(self.input_fname))
        file_content = utils.load_file_content(self.input_fname)

        print("Converting")
        for key in self.keys_to_convert:
            file_content[key] = self.convert(file_content[key])

        print("Writing output_file to {}".format(self.output_fname))
        utils.write_content(self.output_fname, file_content)

    def convert(self, data):
        if self.output_format == "xfel":
            return utils.convert_to_xfel_format(self.channel, data)
        elif self.output_format == "agipd":
            return utils.convert_to_agipd_format(self.channel, data)
        else:
            msg = "Format to which data should be converted is not supported."
            raise Exception(msg)


if __name__ == "__main__":
    args = get_arguments()

    base_dir = args.base_dir
    input_filename = args.input_file
    output_filename = args.output_file
    channel = args.channel
    output_format = args.output_format


#    base_dir = "/gpfs/exfel/exp/SPB/201730/p900009/scratch/user/kuhnm/pcdrs"
#    channel = 0
#
#    input_filename = "pcdrs_AGIPD{:02d}_agipd.h5"
    input_fname = os.path.join(base_dir, input_filename.format(channel))
#
#    #output_format = "agipd"
#    output_format = "xfel"
#
#    output_filename = "{}_AGIPD{:02d}_{}.h5".format("pcdrs")
    output_fname = os.path.join(base_dir,
                                output_filename.format(channel, output_format))

    obj = ConvertFormat(input_fname, output_fname, output_format, channel)
    obj.run()
