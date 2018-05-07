import glob
import h5py
import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # noqa E402
import numpy as np
import os

class PlotBase():
    def __init__(self, input_fname, output_dir, row, col, memcell):

        self._input_fname = input_fname
        self._output_dir = os.path.normpath(output_dir)
        self._row = row
        self._col = col
        self._memcell = memcell

        self._paths = {
            "analog": "analog",
            "digital": "digital",
        }

        self.scaling_point = 200
        self.scaling_factor = 10

        self._data = self._load_data()

    def create_dir(self):
        if not os.path.exists(self._output_dir):
            print("Output directory {} does not exist. Create it."
                  .format(self._output_dir))
            os.makedirs(self._output_dir)

    def _load_data(self):
        data = {}
        with h5py.File(self._input_fname, "r") as f:
            for key, path in self._paths.items():
                data[key] = f[path][:, self._memcell, self._row, self._col]

        return data

    def _scale_full_x_axis(self, number_of_x_values):

        lower = np.arange(self.scaling_point)
        upper = (np.arange(self.scaling_point, number_of_x_values)
                 * self.scaling_factor
                 - self.scaling_point * self.scaling_factor
                 + self.scaling_point)

        scaled_x_values = np.concatenate((lower, upper))

        return scaled_x_values

    def _generate_single_plot(self, x, data, plot_title, label, out_fname, nbins):
        print("_generate_singl_plot method is not implemented.")

    def plot(self):
        self.create_dir()

        plot_title = ("raw data Pixel=[{}, {}], Memcell={:03}"
                      .format(self._row, self._col, self._memcell))
        out_fname = ("raw_data_[{}, {}]_{:03}"
                     .format(self._row, self._col, self._memcell))

        out = os.path.join(self._output_dir, out_fname)

        print("generate plot:", self._row, self._col, self._memcell)
        self._generate_single_plot(plot_title=plot_title,
                                   out_fname=out)
