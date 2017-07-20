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
        # in how many the base interval should be split
        self.n_intervals = {
            "high": 1,
            "medium": 1,
            "low": 5
        }
        self.fit_cutoff_left = 1
        self.fit_cutoff_right = 1

        self.pixel = None
        self.mem_cell = None
        self.stage_idx = None
        self.scaled_x_values = None
        self.hist = None
        self.bin = None
        self.coefficient_matrix = None
        self.x_values = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.data_to_fit = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.slope = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.offset = {
            "high" : [[] for _ in np.arange(self.n_intervals["high"])],
            "medium": [[] for _ in np.arange(self.n_intervals["medium"])],
            "low": [[] for _ in np.arange(self.n_intervals["low"])]
        }
        self.thresholds = None

        self.result_slope = None
        self.result_offset = None
        self.result_thresholds = None

        plot_prefix = "M314_itestc150_asic1"
        if plot_dir is not None:
            plot_prefix = os.path.join(self.plot_dir, plot_prefix)

        self.origin_data_plot_name = Template(plot_prefix + "_${px}_${mc}_data")
        self.fitting_plot_name = Template(plot_prefix + "_${px}_${mc}_fit")
        self.combined_plot_name = Template(plot_prefix + "_${px}_${mc}_combined")
        self.colormap_plot_name = Template(plot_prefix + "_${mc}_colormap")
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

    def run(self, pixel_v_list, pixel_u_list, mem_cell_list,
            create_error_plots, create_colormaps, write_data=True):

        # initiate
        # +1 because counting starts with zero
        shape_tmp = (pixel_v_list.max() + 1, pixel_u_list.max() + 1, mem_cell_list.max() + 1)
        shape = (self.n_gain_stages, ) + shape_tmp
        # thresholds between to distinguish between the gain stages
        threshold_shape = (self.n_gain_stages - 1,) + shape_tmp
        print("result shape: {}".format(shape))
        print("threshold shape: {}".format(threshold_shape))

        self.result_slope = np.zeros(shape, np.float32)
        self.result_offset = np.zeros(shape, np.float32)
        self.result_thresholds = np.zeros(threshold_shape, np.float32)

        self.result_slope_tmp = {
            "high": np.zeros(shape_tmp + (self.n_intervals["high"],)),
            "medium": np.zeros(shape_tmp + (self.n_intervals["low"],)),
            "low": np.zeros(shape_tmp + (self.n_intervals["low"],)),
        }
        self.result_offset_tmp = {
            "high": np.zeros(shape_tmp + (self.n_intervals["high"],)),
            "medium": np.zeros(shape_tmp + (self.n_intervals["low"],)),
            "low": np.zeros(shape_tmp + (self.n_intervals["low"],)),
        }

        for pixel_v in pixel_v_list:
            for pixel_u in pixel_u_list:
                for mem_cell in mem_cell_list:
                    try:
                        self.pixel = [pixel_v, pixel_u]
                        self.mem_cell = mem_cell
                        self.current_idx = (pixel_v, pixel_u, mem_cell)
                        stage_idx = {
                            "high": (0, ) + self.current_idx,
                            "medium": (1, ) + self.current_idx,
                            "low": (2, ) + self.current_idx
                        }

                        self.process_data_point(self.pixel, self.mem_cell)

                        for gain in stage_idx:
                            self.result_slope_tmp[gain][self.current_idx] = self.slope[gain]
                            self.result_offset_tmp[gain][self.current_idx] = self.offset[gain]

                            l = self.fit_cutoff_left
                            r = self.fit_cutoff_right
                            if len(self.slope[gain]) >= l + r + 1:
                                self.result_slope[stage_idx[gain]] = np.mean(self.slope[gain][l:-r])
                                self.result_offset[stage_idx[gain]] = np.mean(self.offset[gain][l:-r])
                            else:
                                self.result_slope[stage_idx[gain]] = np.mean(self.slope[gain])
                                self.result_offset[stage_idx[gain]] = np.mean(self.offset[gain])

                        # store the thresholds
                        for i in np.arange(len(self.thresholds)):
                            self.result_thresholds[(i, ) + self.current_idx] = self.thresholds[i]

                    except KeyboardInterrupt:
                        sys.exit(1)
                    except Exception as e:
                        print("Failed to run for pixel {} and mem_cell {}"
                              .format(self.pixel, self.mem_cell))
                        print(traceback.format_exc())

                        if create_error_plots:
                            try:
                                self.generate_data_plot()
                            except:
                                print("Failed to generate plot")
                                raise

                        #raise

        if write_data:
            self.write_data()

        if create_colormaps:
            if type(create_colormaps) == list and "matrix" in create_colormaps:
                self.generate_colormap_matrix()
            else:
                self.generate_colormap_matrix()
                self.plot_colormap()


    def process_data_point(self, pixel, mem_cell):
        self.hist, self.bins = np.histogram(self.digital[self.pixel[0], self.pixel[1],
                                                         self.mem_cell, :],
                                            bins=self.nbins)
        #print("hist={}".format(self.hist))
        #print("bins={}".format(self.bins))

        self.calc_thresholds()

        self.fit_gain("high", 0, self.thresholds[0], cut_off_ends=False)
        self.fit_gain("medium", self.thresholds[0], self.thresholds[1], cut_off_ends=False)
        self.fit_gain("low", self.thresholds[1], None, cut_off_ends=False)

        if self.create_plot_type:
            print("\nGenerate plots")
            t = time.time()

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

            print("took time: {}".format(time.time() - t))

    def calc_thresholds(self):
        intervals = contiguous_regions(self.hist < self.thold_for_zero)

        # remove the first interval found if it starts with the first bin
        if intervals[0][0] == 0:
            intervals = intervals[1:]
            #print("Pixel {}, mem cell {}: Removed first interval".format(self.pixel, self.mem_cell))

        # remove the last interval found if it ends with the last bin
        if intervals[-1][1] == self.nbins:
            intervals = intervals[1:]
            #print("Pixel {}, mem cell {}: Removed last interval".format(self.pixel, self.mem_cell))

        if len(intervals) < 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("intervals={}".format(intervals))
            raise Exception("Too few intervals")
        if len(intervals) > 2:
            print("thold_for_zero={}".format(self.thold_for_zero))
            print("intervals={}".format(intervals))
            raise Exception("Too many intervals")

        mean_zero_region = np.mean(intervals, axis=1).astype(int)
        #print("mean_zero_region={}".format(mean_zero_region))

        self.thresholds = self.bins[mean_zero_region]
        #print("thesholds={}".format(self.thresholds))

    def determine_fit_interval(self, threshold_l, threshold_u):
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
        return contiguous_regions(condition)

    def split_interval(self, interval, nsplits):
        """
        split given interval into equaly sized sub-intervals
        """
        #print("nsplits", nsplits)
        #print("interval={}".format(interval))

        step = int((interval[-1] - interval[0]) / nsplits)

        splitted_intervals = []
        for i in np.arange(nsplits):
            splitted_intervals.append(
                [interval[0] + i * step, interval[0] + (i + 1) * step])

        # due to rounding or int convertion the last calculated interval ends
        # before the end of the given interval
        splitted_intervals[-1][1] = interval[1]

        #print("splitted_intervals={}".format(splitted_intervals))

        return splitted_intervals

    def fit_gain(self, gain, threshold_l, threshold_u, cut_off_ends):
#        print("\nfitting {} gain with threshholds ({}, {})".format(gain, threshold_l, threshold_u))
        interval_tmp = self.determine_fit_interval(threshold_l, threshold_u)
        #TODO: verify if this is really the right approach
        # sometimes there is more than one interval returned, e.g. [[ 370,  372],[ 373, 1300]]
        # combine it
        interval = [interval_tmp[0][0], interval_tmp[-1][-1]]
        # the interval is splitted into sub-intervals
        intervals = self.split_interval(interval, self.n_intervals[gain])

        for i in np.arange(len(intervals)):
            self.fit_data(gain, intervals[i], i, cut_off_ends=cut_off_ends)

    def fit_data(self, gain, interval, interval_idx, cut_off_ends=False):

        # find the inner points (cut off the top and bottom part
        if cut_off_ends:
            tmp = np.arange(interval[0], interval[1])
            # 100.0 is needed because else it casts it as ints, i.e. 10/100=>0
            lower_border = int(tmp[0] + len(tmp) * self.percent/100.0)
            upper_border = int(tmp[0] + len(tmp) * (1 - self.percent/100.0))
            #print("lower_border = {}".format(lower_border))
            #print("upper_border = {}".format(upper_border))
        else:
            lower_border = interval[0]
            upper_border = interval[1]

        # transform the problem y = mx + c
        # into the form y = Ap, where A = [[x 1]] and p = [[m], [c]]
        # meaning  data_to_fit = coefficient_matrix * [slope, offset]
        self.data_to_fit[gain][interval_idx] = self.analog[self.pixel[0], self.pixel[1],
                                                           self.mem_cell,
                                                           lower_border:upper_border]

        self.x_values[gain][interval_idx] = np.arange(lower_border, upper_border)

        # scaling
        self.scale_x_interval(gain, interval_idx)

        # .T means transposed
        self.coefficient_matrix = np.vstack([self.x_values[gain][interval_idx],
                                             np.ones(len(self.x_values[gain][interval_idx]))]).T

        # fit the data
        # reason to use numpy lstsq:
        # https://stackoverflow.com/questions/29372559/what-is-the-difference-between-numpy-linalg-lstsq-and-scipy-linalg-lstsq
        #lstsq returns: Least-squares solution (i.e. slope and offset), residuals, rank, singular values
        try:
            self.slope[gain][interval_idx], self.offset[gain][interval_idx] = (
                np.linalg.lstsq(self.coefficient_matrix, self.data_to_fit[gain][interval_idx])[0])
        except:
            print("interval\n{}".format(interval))
            print("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
            print("self.data_to_fit[{}][{}]\n{}"
                  .format(gain, interval_idx, self.data_to_fit[gain][interval_idx]))
            raise

        #print("found slope: {}".format(self.slope[gain]))
        #print("found offset: {}".format(self.offset[gain]))

    def scale_full_x_axis(self):
        lower  = np.arange(self.scaling_point)
        upper = (np.arange(self.scaling_point, self.digital.shape[3])
                 * self.scaling_factor
                 - self.scaling_point * self.scaling_factor
                 + self.scaling_point)

        self.scaled_x_values = np.concatenate((lower, upper))

    def scale_x_interval(self, gain, interval_idx):
        # tmp variables to improve reading
        indices_to_scale = np.where(self.x_values[gain][interval_idx] > self.scaling_point)
        x = self.x_values[gain][interval_idx][indices_to_scale]
        # shift x value to root, scale, shift back
        # e.g. x_new = (x_old - 200) * 10 + 200
        self.x_values[gain][interval_idx][indices_to_scale] = (
            (x - self.scaling_point) * self.scaling_factor + self.scaling_point)

    def write_data(self):
        save_file = h5py.File(self.output_fname, "w", libver="latest")

        try:
            print("\nStart saving data")
            t = time.time()

            save_file.create_dataset("/slope", data=self.result_slope)
            save_file.create_dataset("/offset", data=self.result_offset)
            save_file.create_dataset("/thresholds", data=self.result_thresholds)

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

    def remove_legend_dubplicates(self):
        # Remove duplicates in legend
        # https://stackoverflow.com/questions/26337493/pyplot-combine-multiple-line-labels-in-legend
        handles, labels = plt.gca().get_legend_handles_labels()
        i =1
        while i<len(labels):
            if labels[i] in labels[:i]:
                del(labels[i])
                del(handles[i])
            else:
                i +=1

        return handles, labels


    def generate_fit_plot(self, gain):
        # plot fitting
        fig = plt.figure(figsize=None)
        for i in np.arange(len(self.x_values[gain])):
            plt.plot(self.x_values[gain][i],
                     self.data_to_fit[gain][i],
                     'o', color="C0", label="Original data", markersize=1)

            plt.plot(self.x_values[gain][i],
                     self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                     "r", label="Fitted line ({} intervals)".format(self.n_intervals[gain]))

        handles, labels = self.remove_legend_dubplicates()
        plt.legend(handles, labels)

        prefix = self.fitting_plot_name.substitute(px=self.pixel,
                                                   # fill number to be of 3 digits
                                                   mc=str(self.mem_cell).zfill(3))
        fig.savefig("{}_{}{}".format(prefix, gain, self.plot_ending))
        fig.clf()
        plt.close(fig)

    def generate_combined_plot(self):
        # plot combined
        fig = plt.figure(figsize=None)
        plt.plot(self.scaled_x_values,
                 self.analog[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="analog")

        plt.plot(self.scaled_x_values,
                 self.digital[self.pixel[0], self.pixel[1], self.mem_cell, :],
                 ".", markersize=0.5, label="digital")

        for gain in ["high", "medium", "low"]:
            # if there are multiple sub-fits some are cut off
            if len(self.x_values[gain]) >= self.fit_cutoff_left + self.fit_cutoff_right + 1:
                # display the cut off fits on the left side
                for i in np.arange(self.fit_cutoff_left):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", alpha=0.3, label="Fitted line (unused)")
                # display the used fits
                for i in np.arange(self.fit_cutoff_left, len(self.x_values[gain]) - self.fit_cutoff_right):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", label="Fitted line")
                # display the cut off fits on the right side
                for i in np.arange(len(self.x_values[gain]) - self.fit_cutoff_right,
                                   len(self.x_values[gain])):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", alpha=0.3, label="Fitted line (unused)")
            else:
                # display all fits
                for i in np.arange(len(self.x_values[gain])):
                    plt.plot(self.x_values[gain][i],
                             self.slope[gain][i] * self.x_values[gain][i] + self.offset[gain][i],
                             "r", label="Fitted line")

        handles, labels = self.remove_legend_dubplicates()
        plt.legend(handles, labels)

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

    def generate_colormap_matrix(self):
        """
        generate map for low gain
        """
        gain = 2
        gain_name = "low"
        self.colormap_matrix_slope = None
        self.colormap_matrix_offset = None

        mem_cell = 1
        l = self.fit_cutoff_left
        r = self.fit_cutoff_right
        n_used_fits = self.n_intervals["low"] - l - r

        # gain, pixel_v, pixel_u, mem_cell
        n_pixel_v, n_pixel_u = self.result_slope.shape[1:3]

        # create array
        self.colormap_matrix_slope = np.empty((n_pixel_v, n_pixel_u))
        # intiate with nan
        self.colormap_matrix_slope[:, :] = np.NAN

        # create array
        self.colormap_matrix_offset = np.empty((n_pixel_v, n_pixel_u))
        # intiate with nan
        self.colormap_matrix_offset[:, :] = np.NAN

        for v in np.arange(n_pixel_v):
            for u in np.arange(n_pixel_u):
                #print("pixel [{},{}], mem_cell {}".format(v, u, mem_cell))

                mean = self.result_slope[2, v, u, mem_cell]
#                print(self.result_slope_tmp["low"])
#                print(self.result_slope_tmp["low"][v, u, mem_cell])
                values = self.result_slope_tmp["low"][v, u, mem_cell][l:-r]
                #print("slope")
                #print("mean={}, values={}".format(mean, values))

                try:
                    if mean != 0:
                        self.colormap_matrix_slope[v, u] = (
                            np.linalg.norm(values - mean)/(n_used_fits * mean))
                except:
                    print("pixel {}, mem_cell {}".format(v, u, mem_cell))
                    print("slope")
                    print("mean={}".format(mean))
                    print("values={}".format(values))


                mean = self.result_offset[2, v, u, mem_cell]
                values = self.result_offset_tmp["low"][v, u, mem_cell][l:-r]

                try:
                    if mean != 0:
                        self.colormap_matrix_offset[v, u] = (
                            np.linalg.norm(values - mean)/(n_used_fits * mean))
                except:
                    print("pixel {}, mem_cell {}".format(v, u, mem_cell))
                    print("slope")
                    print("mean={}".format(mean))
                    print("values={}".format(values))

                #print("colormap_matrix_slopes={}".format(self.colormap_matrix_slopes[v, u]))

        #print("colormap")
        #print(colormap_matrix_slopes)

    def plot_colormap(self):
        # fill number to be of 3 digits
        prefix = self.colormap_plot_name.substitute(mc=str(self.mem_cell).zfill(3))

        fig = plt.figure(figsize=None)
        # Convert it to a masked array instead of just using nan's..
        #self.colormap_matrix_slope = np.ma.masked_invalid(self.colormap_matrix_slope)
        plt.imshow(self.colormap_matrix_slope)
        plt.colorbar()
        #bad_data = np.ma.masked_where(~colormap_matrix_slope.mask, colormap_matrix_slope.mask)
        #plt.imshow(bad_data, interpolation='nearest', cmap=plt.get_cmap("Reds"))

        fig.savefig("{}_{}_{}{}".format(prefix, "low", "slope", self.plot_ending))
        fig.clf()
        plt.close(fig)

#        fig = plt.figure(figsize=None)
#        plt.hist(self.colormap_matrix_slope.flatten(), bins=300)

        #plt.show()
#        fig.savefig("{}_{}_{}_hist{}".format(prefix, "low", "slope", self.plot_ending))
#        fig.clf()
#        plt.close(fig)

        fig = plt.figure(figsize=None)
        plt.imshow(self.colormap_matrix_offset)
        plt.colorbar()

        fig.savefig("{}_{}_{}{}".format(prefix, "low", "offset", self.plot_ending))
        fig.clf()
        plt.close(fig)



if __name__ == "__main__":

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"

    input_file = os.path.join(base_dir, "M314/temperature_m15C/drscs/itestc150/M314_drscs_itestc150_asic01.h5")
    output_file = os.path.join(base_dir, "M314/temperature_m15C/drscs/itestc150/M314_drscs_itestc150_asic01_results.h5")
    plot_dir = os.path.join(base_dir, "M314/temperature_m15C/drscs/plots/itestc150")

    create_error_plots = True
    #pixel_v_list = np.arange(0, 1)
    #pixel_u_list = np.arange(0, 32)
    pixel_v_list = np.arange(64)
    pixel_u_list = np.arange(64)
    mem_cell_list = np.arange(1, 2)

    #create_plots can be set to False, "data", "fit", "combined" or "all"
    cal = ProcessDrscs(input_file, output_file, plot_dir=plot_dir, create_plots=False)

    print("\nRun processing")
    t = time.time()
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list, create_error_plots, create_colormaps=False)
    print("Processing took time: {}".format(time.time() - t))
