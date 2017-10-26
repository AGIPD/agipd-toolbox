import os
import h5py

import utils


class JoinConstants():
    def __init__(self):
        self.base_dir = "/home/kuhnm/ownCloud/calibration_data_example"
        self.input_fname = "Pedestal_in_ADU_m3_P11burst1k_nofit.hdf5"

        self.source_content = None

        self.run()

    def run(self):

        fname = "/home/kuhnm/tmp/test/test.h5"

        f = None
        try:
            f = h5py.File(fname, "w")

            for channel in range(16):
                fname = os.path.join(self.base_dir, self.input_fname)

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
    JoinConstants()
