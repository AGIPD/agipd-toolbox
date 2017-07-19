import matplotlib.pyplot as plt
import numpy as np
import h5py
from string import Template
import os
import sys
import time
import traceback

# from Joe Kington's answer here https://stackoverflow.com/a/4495197/3751373
def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition, n=1)
#    print("d", d)
    idx = d.nonzero()[0]
#    print("idx", idx)

    # output of np.nonzero is tuple -> convert to np.array
    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right. -JK
    idx += 1
#    print("idx", idx)

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]
#    print("start of cond: idx", idx)

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit
#    print("end of cond: idx", idx)

    # Reshape the result into two columns
    idx.shape = (-1,2)
#    print("idx", idx)
    return idx

class ProcessDrscs():
    def __init__(self, input_file_name, output_file_name, plot_dir=None, create_plots=False):

        self.input_fname = input_file_name
        self.output_fname = output_file_name

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.create_plot_type = create_plots
        self.plot_dir = plot_dir

        self.digital = None
        self.analog = None

        self.percent = 10
        self.nbins = 30

        # threshold for regions which are concidered to containing no gain information
        self.thold_for_zero = 4

        self.scaling_point = 200
        self.scaling_factor = 10

        self.n_gain_stages = 3

        self.pixel = None
        self.mem_cell = None
        self.stage_idx = None
        self.scaled_x_values = None
        self.hist = None
        self.bin = None
        self.coefficient_matrix = None
        self.x_values = {
            "high" : None,
            "medium": None,
            "low": None
        }
        self.data_to_fit = {
            "high" : None,
            "medium": None,
            "low": None
        }
        self.slope = None
        self.offset = None

        plot_prefix = "M314_itestc150_asic1"
        if plot_dir is not None:
            plot_prefix = os.path.join(self.plot_dir, plot_prefix)

        self.origin_data_plot_name = Template(plot_prefix + "_${px}_${mc}_data")
        self.fitting_plot_name = Template(plot_prefix + "_${px}_${mc}_fit")
        self.combined_plot_name = Template(plot_prefix + "_${px}_${mc}_combined")
        self.plot_ending = ".png"

        print("Load data")
        t = time.time()
        self.load_data()
        print("took time: {}".format(time.time() - t))
        self.scale_full_x_axis()

    def load_data(self):

        source_file = h5py.File(self.input_fname, "r")

        # origin data is written as int16 which results in a integer overflow
        # when handling the scaling
        self.analog = source_file[self.analog_path][()].astype("int32")
        self.digital = source_file[self.digital_path][()].astype("int32")

        source_file.close()

    def run(self, pixel_v_list, pixel_u_list, mem_cell_list, create_error_plots):

        # initiate
        # +1 because counting starts with zero
        shape = (self.n_gain_stages, pixel_v_list.max() + 1, pixel_u_list.max() + 1, mem_cell_list.max() + 1)
        # thresholds between to distinguish between the gain stages
        threshold_shape = (self.n_gain_stages - 1,
                           pixel_v_list.max() + 1,
                           pixel_u_list.max() + 1,
                           mem_cell_list.max() + 1)
        print("result shape: {}".format(shape))
        print("theshold shape: {}".format(shape))

        self.slope = np.zeros(shape, np.float32)
        self.offset = np.zeros(shape, np.float32)
        self.thresholds = np.zeros(threshold_shape, np.float32)

        for pixel_v in pixel_v_list:
            for pixel_u in pixel_u_list:
                for mem_cell in mem_cell_list:
                    try:
                        self.pixel = [pixel_v, pixel_u]
                        self.mem_cell = mem_cell
                        self.current_idx = (pixel_v, pixel_u, mem_cell)
                        self.stage_idx = {
                            "high": (0, ) + self.current_idx,
                            "medium": (1, ) + self.current_idx,
                            "low": (2, ) + self.current_idx
                        }

                        self.process_data_point(self.pixel, self.mem_cell)
                    except KeyboardInterrupt:
                        sys.exit(1)
                    except Exception as e:
                        print("Failed to run for pixel {} and mem_cell {}"
                              .format(self.pixel, self.mem_cell))
                        print("Error was: {}".format(e))
                        print(traceback.print_exc())

                        if create_error_plots:
                            try:
                                self.generate_data_plot()
                            except:
                                print("Failed to generate plot")
                                raise

                        #raise

        self.write_data()

    def process_data_point(self, pixel, mem_cell):
        self.hist, self.bins = np.histogram(self.digital[self.pixel[0], self.pixel[1],
                                                         self.mem_cell, :],
                                            bins=self.nbins)
        #print("hist={}".format(self.hist))
        #print("bins={}".format(self.bins))

        self.calc_thresholds()
        threshold = self.thresholds[:, self.current_idx[0], self.current_idx[1], self.current_idx[2]]

#        print("\nfitting high gain with thresholds ({}, {})".format(0, threshold[0]))
        self.fit_data(0, threshold[0], "high")

        #print("\nfitting medium gain")
        self.fit_data(threshold[0], threshold[1], "medium")

#        print("\nfitting low gain with threshholds ({}, {})".format(threshold[1], None))
        self.fit_data(threshold[1], None, "low")

        if self.create_plot_type:
            if "data" in self.create_plot_type:
                self.generate_data_plot()

            elif "fit" in self.create_plot_type:
                self.generate_fit_plot("high")
                self.generate_fit_plot("medium")
                self.generate_fit_plot("low")

            elif "combined" in self.create_plot_type:
                self.generate_combined_plot()

            else:
                self.generate_all_plots()


    def calc_thresholds(self):
        idx = contiguous_regions(self.hist < self.thold_for_zero)

        # remove the first intervall found if it starts with the first bin
        if idx[0][0] == 0:
            idx = idx [1:]
            #print("Pixel {}, mem cell {}: Removed first intervall".format(self.pixel, self.mem_cell))

        # remove the last intervall found if it ends with the last bin
        if idx[-1][1] == self.nbins:
            idx = idx [1:]
            #print("Pixel {}, mem cell {}: Removed last intervall".format(self.pixel, self.mem_cell))

        if len(idx) < 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("idx={}".format(idx))
            raise Exception("Too few intervalls")
        if len(idx) > 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("idx={}".format(idx))
            raise Exception("Too many intervalls")

        mean_zero_region = np.mean(idx, axis=1).astype(int)
        #print("mean_zero_region={}".format(mean_zero_region))

        #self.threshold = self.bins[mean_zero_region]
        tmp = self.bins[mean_zero_region]
        for i in xrange(len(tmp)):
            self.thresholds[(i, ) + self.current_idx] = tmp[i]

        #print("theshold={}".format(self.threshold))

    def fit_data(self, threshold_l, threshold_u, gain):
        data_d = self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :]

        # determine condition
        if threshold_l is not None and threshold_u is not None:
            # the & will give you an elementwise and (the parentheses are necessary).
            condition = (threshold_l < data_d) & (data_d < threshold_u)
            #print("condition = ({} < data_d) & (data_d < {})".format(threshold_l, threshold_u))
        elif threshold_l is not None:
            condition = threshold_l < data_d
            #print("condition = {} < data_d".format(threshold_l))
        elif threshold_u is not None:
            condition = data_d < threshold_u
            #print("condition = data_d < {}".format(threshold_u))

        #print("condition value={}".format(condition))
        idx = contiguous_regions(condition)[0]

        # find the inner points (cut off the top and bottom part
        tmp = np.arange(idx[0], idx[1])
        lower_border = int(tmp[0] + len(tmp) * self.percent/100)
        upper_border = int(tmp[0] +  len(tmp) * (1 - self.percent/100))
        #print("lower_border = {}".format(lower_border))
        #print("upper_border = {}".format(upper_border))

        # transform the problem y = mx + c
        # into the form y = Ap, where A = [[x 1]] and p = [[m], [c]]
        # meaning  data_to_fit = coefficient_matrix * [slope, offset]
        self.data_to_fit[gain] = self.analog[self.pixel[0], self.pixel[1],
                                             self.mem_cell,
                                             lower_border:upper_border]

        self.x_values[gain] = np.arange(lower_border, upper_border)

        # scaling
        self.scale_x_axis_intervall(gain)

        self.coefficient_matrix = np.vstack([self.x_values[gain],
                                             np.ones(len(self.x_values[gain]))]).T

        # fit the data
        idx = self.stage_idx[gain]
        # reason to use numpy lstsq:
        # https://stackoverflow.com/questions/29372559/what-is-the-difference-between-numpy-linalg-lstsq-and-scipy-linalg-lstsq
        #lstsq returns: Least-squares solution (i.e. slope and offset), residuals, rank, singular values
        self.slope[idx], self.offset[idx] = np.linalg.lstsq(self.coefficient_matrix,
                                                            self.data_to_fit[gain])[0]

        #self.slope[gain], self.offset[gain] = np.linalg.lstsq(self.coefficient_matrix,
        #                                                      self.data_to_fit[gain])[0]
        #print("found slope: {}".format(self.slope[gain]))
        #print("found offset: {}".format(self.offset[gain]))

    def scale_full_x_axis(self):
        lower  = np.arange(self.scaling_point)
        upper = (np.arange(self.scaling_point, self.digital.shape[3])
                 * self.scaling_factor
                 - self.scaling_point * self.scaling_factor
                 + self.scaling_point)

        self.scaled_x_values = np.concatenate((lower, upper))

    def scale_x_axis_intervall(self, gain):
        # tmp variables to improve reading
        indices_to_scale = np.where(self.x_values[gain] > self.scaling_point)
        x = self.x_values[gain][indices_to_scale]
        # shift x value to root, scale, shift back
        # e.g. x_new = (x_old - 200) * 10 + 200
        self.x_values[gain][indices_to_scale] = (
            (x - self.scaling_point) * self.scaling_factor + self.scaling_point)

    def write_data(self):
        save_file = h5py.File(self.output_fname, "w", libver="latest")

        try:
            print("\nStart saving data")
            t = time.time()

            save_file.create_dataset("/slope", data=self.slope)
            save_file.create_dataset("/offset", data=self.offset)
            save_file.create_dataset("/thresholds", data=self.thresholds)

            save_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            save_file.close()

    def generate_data_plot(self):

        width = 0.7 * (self.bins[1] - self.bins[0])
        center = (self.bins[:-1] + self.bins[1:]) / 2

        # plot data
        fig = plt.figure(figsize=(13, 5))
        ax = fig.add_subplot(121)
        ax.bar(center, self.hist, align="center", width=width, label="hist digital")
        ax.legend()

        ax = fig.add_subplot(122)
        ax.plot(self.scaled_x_values,
                self.analog[self.pixel[0], self.pixel[1], self.mem_cell, :],
                ".", markersize=0.5, label="analog")
        ax.plot(self.scaled_x_values,
                self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :],
                ".", markersize=0.5, label="digital")

        ax.legend()
        prefix = self.origin_data_plot_name.substitute(px=self.pixel,
                                                       # fill number to be of 3 digits
                                                       mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}{}".format(prefix, self.plot_ending))
        fig.clf()
        plt.close(fig)


    def generate_fit_plot(self, gain):
        # plot fitting
        fig = plt.figure(figsize=None)
        plt.plot(self.x_values[gain],
                 self.data_to_fit[gain],
                 'o', label="Original data", markersize=1)

        plt.plot(self.x_values[gain],
                 self.slope[self.stage_idx[gain]] * self.x_values[gain] + self.offset[self.stage_idx[gain]],
                 "r", label="Fitted line high")

        plt.legend()
        prefix = self.fitting_plot_name.substitute(px=self.pixel,
                                                   # fill number to be of 3 digits
                                                   mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}_{}{}".format(prefix, gain, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def generate_combined_plot(self):
        i = self.stage_idx
        # plot combined
        fig = plt.figure(figsize=None)
        plt.plot(self.scaled_x_values,
                 self.analog[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="analog")

        plt.plot(self.scaled_x_values,
                 self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="digital")

        plt.plot(self.x_values["high"],
                 self.slope[i["high"]] * self.x_values["high"] + self.offset[i["high"]],
                 "r", label="Fitted line high")

        plt.plot(self.x_values["medium"],
                 self.slope[i["medium"]] * self.x_values["medium"] + self.offset[i["medium"]],
                 "r", label="Fitted line medium")

        plt.plot(self.x_values["low"],
                 self.slope[i["low"]] * self.x_values["low"] + self.offset[i["low"]],
                 "r", label="Fitted line low")

        plt.legend()
        prefix = self.combined_plot_name.substitute(px=self.pixel,
                                                    # fill number to be of 3 digits
                                                    mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}{}".format(prefix, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def generate_all_plots(self):

        self.generate_data_plot()

        self.generate_fit_plot("high")
        self.generate_fit_plot("medium")
        self.generate_fit_plot("low")

        self.generate_combined_plot()

if __name__ == "__main__":

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"

    input_file = os.path.join(base_dir, "M314/temperature_m15C/drscs/itestc150/M314_drscs_itestc150_asic1.h5")
    output_file = os.path.join(base_dir, "M314/temperature_m15C/drscs/itestc150/M314_drscs_itestc150_asic1_results.h5")
    plot_dir = os.path.join(base_dir, "M314/temperature_m15C/drscs/plots/itestc150")

    create_error_plots = True
    pixel_v_list = np.arange(10,11)
    pixel_u_list = np.arange(11, 12)
    mem_cell_list = np.arange(19, 20)

    #create_plots can be set to False, "data", "fit", "combined" or "all"
    cal = ProcessDrscs(input_file, output_file, plot_dir=plot_dir, create_plots=["combined"])

    print("\nRun processing")
    t = time.time()
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list, create_error_plots)
    print("took time: {}".format(time.time() - t))
