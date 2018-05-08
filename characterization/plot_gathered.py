import matplotlib
matplotlib.use('Agg')  # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt  # noqa E402
import numpy as np

from plot_base import PlotBase

class PlotGathered(PlotBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _generate_single_plot(self, plot_title, out_fname):
        number_of_x_values = self._data["analog"].shape[0]

        x = np.arange(number_of_x_values)
        analog = self._data["analog"]
        digital = self._data["digital"]

        fig = plt.figure(figsize=None)
        plt.plot(x, analog, ".", markersize=0.5, label="analog")
        plt.plot(x, digital, ".", markersize=0.5, label="digital")

        plt.legend()

        fig.suptitle(plot_title)
        fig.savefig(out_fname)

        fig.clf()
        plt.close(fig)
