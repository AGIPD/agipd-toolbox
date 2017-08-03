from __future__ import print_function

from multiprocessing import Pool, TimeoutError
import matplotlib.pyplot as plt
import numpy as np
import h5py
from string import Template
import os
import sys
import time
import traceback
from characterization.plotting import generate_data_plot, generate_fit_plot, generate_combined_plot, generate_all_plots


class IntervalError(Exception):
    pass

class FitError(Exception):
    pass

class IntervalSplitError(Exception):
    pass

class GainStageNumberError(Exception):
    pass

class ThresholdNumberError(Exception):
    pass

class InfiniteLoopNumberError(Exception):
    pass

def check_file_exists(file_name):
    print("save_file = {}".format(file_name))
    if os.path.exists(file_name):
        print("Output file already exists")
        sys.exit(1)
    else:
        print("Output file: ok")

# not defined inside the class to let other classes reuse this
def initiate_result(pixel_v_list, pixel_u_list, mem_cell_list, n_gain_stages,
                    n_intervals, n_diff_changes_stored):

    result = {
        "slope": {
            "mean": None,
            "individual": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "offset": {
            "mean": None,
            "individual": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "residuals": {
            "mean": None,
            "individual": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "average_residual": {
            "mean": None,
            "individual": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "intervals": {
            "gain_stages": None,
            "subintervals": {
                "high" : None,
                "medium": None,
                "low": None
            }
        },
        "medians": {
            "high" : None,
            "medium": None,
            "low": None
        },
        "thresholds": None,
        "error_code": None,
        "warning_code": None,
        "collection": {
            "diff_threshold": None,
            "region_range": None,
            "safty_factor" : None,
            "diff_changes_idx": None,
            "len_diff_changes_idx": None,
            "scaling_point": None,
            "scaling_factor": None,
            "n_gain_stages": None,
            "fit_cutoff_left": None,
            "fit_cutoff_right": None,
        }
    }

    # +1 because counting starts with zero
    shape_tmp = (pixel_v_list.max() + 1,
                 pixel_u_list.max() + 1,
                 mem_cell_list.max() + 1)

    shape = (n_gain_stages, ) + shape_tmp
    print("result shape: {}".format(shape))

    # thresholds between to distinguish between the gain stages
    threshold_shape = (n_gain_stages - 1,) + shape_tmp
    print("threshold shape: {}".format(threshold_shape))

    # initiate fit results
    for key in ["slope", "offset", "residuals", "average_residual"]:
        result[key]["mean"] = np.zeros(shape, np.float32)
        result[key]["individual"] = {
            "high": np.zeros(shape_tmp + (n_intervals["high"],)),
            "medium": np.zeros(shape_tmp + (n_intervals["medium"],)),
            "low": np.zeros(shape_tmp + (n_intervals["low"],)),
        }
    result["medians"] = np.zeros(shape, np.float32)
    # initiated with -1 to distinguish between code failures and catched errors
    result["error_code"] = -1 * np.ones(shape_tmp, np.int16)
    result["warning_code"] = np.zeros(shape_tmp, np.int16)

    # intiate intervals
    # interval consists of start and end point -> x2
    result["intervals"]["gain_stages"] = init_with_nan(shape + (2,), np.int)
    for key in ["high", "medium", "low"]:
        result["intervals"]["subintervals"][key] = (
            init_with_nan(shape_tmp + (n_intervals[key], 2), np.int))

    # initiate thresholds
    result["thresholds"] = np.zeros(threshold_shape, np.float32)

    result["collection"]["diff_changes_idx"] = init_with_nan(
        shape_tmp + (n_diff_changes_stored,), np.int)
    result["collection"]["len_diff_changes_idx"] = np.zeros(shape_tmp, np.int)

    return result

def init_with_nan(shape, dtype):
    # create array
    obj = np.empty(shape, dtype=dtype)
    # intiate with nan
    obj[...] = np.NAN

    return obj

# gives back to N biggest entries of a list
# modification of: https://stackoverflow.com/questions/12787650/finding-the-index-of-n-biggest-elements-in-python-array-list-efficiently
def biggest_entries(list_to_check, N):
    return np.argsort(list_to_check)[-N:]

class ProcessDrscs():
    def __init__(self, asic, input_fname=False, output_fname=False,
                 analog=None, digital=None,
                 plot_prefix=None, plot_dir=None, create_plots=False):

        if input_fname:
            self.input_fname = input_fname
            self.analog = None
            self.digital = None
        else:
            self.input_fname = None
            self.analog = analog
            self.digital = digital

        if output_fname:
            self.output_fname = output_fname
        else:
            self.output_fname = None

        self.digital_path = "/entry/instrument/detector/data_digital"
        self.analog_path = "/entry/instrument/detector/data"

        self.create_plot_type = create_plots
        self.plot_dir = plot_dir

        self.percent = 10

        self.diff_threshold = -100
        self.region_range = 10
        self.safty_factor = 1000
        self.n_diff_changes_stored = 10


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

        self.use_digital = False

        self.gain_idx = None
        self.scaled_x_values = None
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

        ###
        # error_code:
        # -1   Not handled (something went wrong)
        # 1    Unknown error
        # 2    Too many gain stages found
        # 3    Not enough gain stages found
        # 4    Breaking infinite loop
        #
        # warning_code:
        # 2    Use then two biggest intervals
        ###


        if plot_prefix:
            print("plot_prefix", plot_prefix)
            self.plot_file_prefix = "{}_asic{}".format(plot_prefix, str(asic).zfill(2))
            if plot_dir is not None:
                self.plot_file_prefix = os.path.join(self.plot_dir, self.plot_file_prefix)

            self.plot_title_prefix = self.plot_file_prefix.rsplit("/", 1)[1]
            print("plot_file_prefix", self.plot_file_prefix)
            print("plot_title_prefix", self.plot_title_prefix)

            self.plot_ending = ".png"
        else:
            self.plot_file_prefix = None
            self.plot_title_prefix = None
            self.plot_ending = None

        if self.output_fname is not None:
            check_file_exists(self.output_fname)

        if self.input_fname is not None:
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
            create_error_plots=False):

        if len(pixel_v_list) == 0 or len(pixel_u_list) == 0:
            print("pixel_v_list", pixel_v_list)
            print("pixel_u_list", pixel_u_list)
            raise Exception("No proper pixel specified")
        if len(mem_cell_list) == 0:
            print("mem_cell_list", mem_cell_list)
            raise Exception("No proper memory cell specified")

        if create_error_plots and self.plot_title_prefix is None:
            raise Exception("Plotting was not defined on initiation. Quitting.")

        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list

        # initiate
        self.result = initiate_result(self.pixel_v_list, self.pixel_u_list,
                                      self.mem_cell_list, self.n_gain_stages,
                                      self.n_intervals, self.n_diff_changes_stored)

        self.result["collection"]["diff_threshold"] = self.diff_threshold
        self.result["collection"]["region_range"] = self.region_range
        self.result["collection"]["safty_factor"] = self.safty_factor
        self.result["collection"]["n_diff_changes_stored"] = self.n_diff_changes_stored
        self.result["collection"]["scaling_point"] = self.scaling_point
        self.result["collection"]["scaling_factor"] = self.scaling_factor
        self.result["collection"]["n_gain_stages"] = self.n_gain_stages
        self.result["collection"]["fit_cutoff_left"] = self.fit_cutoff_left
        self.result["collection"]["fit_cutoff_right"] = self.fit_cutoff_right

        for pixel_v in self.pixel_v_list:
            for pixel_u in self.pixel_u_list:
                #t = time.time()
                for mem_cell in self.mem_cell_list:
                    try:
                        self.current_idx = (pixel_v, pixel_u, mem_cell)
                        self.gain_idx = {
                            "high": (0, ) + self.current_idx,
                            "medium": (1, ) + self.current_idx,
                            "low": (2, ) + self.current_idx
                        }

                        self.process_data_point(self.current_idx)

                        for gain in self.gain_idx:
                            l = self.fit_cutoff_left
                            r = self.fit_cutoff_right
                            if len(self.result["slope"]["individual"][gain][self.current_idx]) >= l + r + 1:
                                for t in ["slope", "offset", "residuals", "average_residual"]:
                                    self.result[t]["mean"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain][self.current_idx][l:-r]))
                            else:
                                for t in ["slope", "offset", "residuals", "average_residual"]:
                                    self.result[t]["mean"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain][self.current_idx]))

                        if self.result["error_code"][self.current_idx] < 0:
                            self.result["error_code"][self.current_idx] = 0

                    except KeyboardInterrupt:
                        sys.exit(1)
                    except Exception as e:
                        if type(e) == IntervalError:
                            #print("IntervalError")
                            pass
                        elif type(e) == IntervalSplitError:
                            #print("IntervalSplitError")
                            pass
                        elif type(e) == GainStageNumberError:
                            #print("{}: GainStageNumberError".format(self.current_idx))
                            #print("found number of diff_changes_idx={}".format(len(self.diff_changes_idx)))
                            pass
                        else:
                            print("Failed to run for pixel [{}, {}] and mem_cell {}"
                                  .format(self.current_idx[0], self.current_idx[1], self.current_idx[2]))
                            print(traceback.format_exc())

                            if self.result["error_code"][self.current_idx] <= 0:
                                self.result["error_code"][self.current_idx] = 1

                        if create_error_plots:
                            try:
                                idx = self.current_idx + (slice(None),)

                                plot_file_prefix = ("{}_[{}, {}]_{}"
                                                    .format(self.plot_file_prefix,
                                                            self.current_idx[0],
                                                            self.current_idx[1],
                                                            str(self.current_idx[2]).zfill(3)))
                                plot_title_prefix = ("{}_[{}, {}]_{}"
                                                     .format(self.plot_title_prefix,
                                                             self.current_idx[0],
                                                             self.current_idx[1],
                                                             str(self.current_idx[2]).zfill(3)))
                                plot_title = "{} data".format(plot_title_prefix)
                                plot_name = "{}_data{}".format(plot_file_prefix,
                                                               self.plot_ending)

                                generate_data_plot(self.current_idx,
                                                   self.scaled_x_values,
                                                   self.analog[idx],
                                                   self.digital[idx],
                                                   plot_title,
                                                   plot_name)
                            except:
                                print("Failed to generate plot")
                                raise

#                        if type(e) not in [IntervalError, IntervalSplitError, GainStageNumberError]:
#                            raise

                #print("Process pixel {} took time: {}".format([pixel_v, pixel_u], time.time() - t))

        if self.output_fname is not None:
            print("writing data")
            self.write_data()

    def process_data_point(self, current_idx):

        data = self.digital[self.current_idx[0], self.current_idx[1], self.current_idx[2], :]

        self.calc_gain_regions()

        self.result["intervals"]["subintervals"]["high"][self.current_idx] = (
            self.fit_gain("high",
                          self.result["intervals"]["gain_stages"][self.gain_idx["high"]],
                          cut_off_ends=False))

        self.result["intervals"]["subintervals"]["medium"][self.current_idx] = (
            self.fit_gain("medium",
                          self.result["intervals"]["gain_stages"][self.gain_idx["medium"]],
                          cut_off_ends=False))

        self.result["intervals"]["subintervals"]["low"][self.current_idx] = (
            self.fit_gain("low",
                          self.result["intervals"]["gain_stages"][self.gain_idx["low"]],
                          cut_off_ends=False))

        self.calc_gain_median()
        self.calc_thresholds()

        if self.create_plot_type:
            print("\nGenerate plots")
            t = time.time()

            self.create_plots()

            print("took time: {}".format(time.time() - t))

    def calc_gain_regions(self):
        data_a = self.analog[self.current_idx[0], self.current_idx[1], self.current_idx[2], :]

        # calculates the difference between neighboring elements
        self.diff = np.diff(data_a)

        self.diff_changes_idx = np.where(self.diff < self.diff_threshold)[0]

        gain_intervals = [[0, 0]]

        #TODO check if diff_changes_idx has to many entries
        #print("current idx: {}", self.current_idx)
        #print(self.diff_changes_idx)
        #for i in self.diff_changes_idx:
        #    print(i, ":", data_a[i:i+2])

        i = 0
        prev_stop = 0
        prev_stop_idx = 0
        pot_start = 0
        set_stop_flag = True
        iteration_check = 0
        last_iteration_borders = []
        while i < len(self.diff_changes_idx):

            if set_stop_flag:
                #print("setting prev_stop")
                prev_stop = self.diff_changes_idx[i]
                prev_stop_idx = i
            # exclude the found point
            pot_start = self.diff_changes_idx[i] + 1

            #print("gain intervals", gain_intervals)
            #print("prev_stop", prev_stop)
            #print("pot_start", pot_start)

            # determine the region before the potention gain stage change
            start_before = prev_stop - self.region_range
            if start_before < 0:
                region_of_interest_before = data_a[0:prev_stop]
            else:
                region_of_interest_before = data_a[start_before:prev_stop]
            #print("region_of_interest_before", region_of_interest_before)

            # determine the region after the potention gain stage change
            stop_after = pot_start + self.region_range
            if stop_after > self.diff.size:
                region_of_interest_after = data_a[pot_start:self.diff.size]
            else:
                region_of_interest_after = data_a[pot_start:stop_after]
            #print("region_of_interest_after", region_of_interest_after)

            # check if the following idx is contained in region_of_interest_before
            near_matches_before = np.where(start_before < self.diff_changes_idx[:prev_stop_idx])
            # np.where returns a tuple (array, type)
            near_matches_before = near_matches_before[0]

            # check if the following idx is contained in region_of_interest_after
            near_matches_after = np.where(self.diff_changes_idx[i + 1:] < stop_after)
            # np.where returns a tuple (array, type)
            near_matches_after = near_matches_after[0]
            #print("near match before", near_matches_before, self.diff_changes_idx[:prev_stop_idx][near_matches_before])
            #print("near match after", near_matches_after, self.diff_changes_idx[i + 1:][near_matches_after])

            if region_of_interest_before.size == 0:
                mean_before = data_a[prev_stop]
            else:
                mean_before = np.mean(region_of_interest_before)

            if region_of_interest_after.size == 0:
                mean_after = data_a[pot_start]
            else:
                mean_after = np.mean(region_of_interest_after)

            #print("mean_before", mean_before)
            #print("mean_after", mean_after)

            if near_matches_before.size == 0 and near_matches_after.size == 0:

                # if the slope is too high the mean is falsified by this
                if (region_of_interest_before.size != 0 and \
                        np.max(region_of_interest_before) - np.min(region_of_interest_before) > self.safty_factor * 2):
                    # cut the region of interest into half
                    mean_before = np.mean(region_of_interest_before[len(region_of_interest_before) / 2:])
                    #print("mean_before after cut down of region of interest:", mean_before)

                if mean_before > mean_after + self.safty_factor:
                    # a stage change was found
                    gain_intervals[-1][1] = prev_stop
                    gain_intervals.append([pot_start, 0])

                i += 1
                set_stop_flag = True
            else:
                if gain_intervals[-1][1] == 0:
                    #print("gain intervals last entry == 0")

                    if near_matches_before.size != 0:
                        #print("near_matches_before is not emtpy")
                        region_start = self.diff_changes_idx[:prev_stop_idx][near_matches_before[-1]] + 1
                        #print("region_start", region_start)

                        region_of_interest_before = data_a[region_start:prev_stop]
                        #print("region_of_interest_before", region_of_interest_before)

                        if region_of_interest_before.size == 0:
                            mean_before = data_a[prev_stop]
                        else:
                            mean_before = np.mean(region_of_interest_before)
                        #print("mean_before", mean_before)

                    if near_matches_before.size == 0:
                        #print("near_matches_before.size == 0")
                        if mean_before > mean_after + self.safty_factor:
                            #print("mean check")
                            gain_intervals[-1][1] = prev_stop

                            # prevent an infinite loop
                            if last_iteration_borders == [prev_stop, pot_start]:
                                if iteration_check >= 10:
                                    self.result["error_code"][self.current_idx] = 4
                                    raise InfiniteLoopNumberError("Breaking infinite loop")
                                iteration_check += 1
                            else:
                                last_iteration_borders = [prev_stop, pot_start]
                                iteration_check = 0

                            i += near_matches_after[-1]
                            set_stop_flag = False
                            continue
                        # for diff changes where there is a jump right after (down, up, stay, stay, jump,...)
                        # cut off the area after the jump and try again
                        elif near_matches_after.size != 0:
                            #print("near_matches_after is not emtpy")
                            region_stop = self.diff_changes_idx[i + 1:][near_matches_after[0]]
                            #print("region_stop", region_stop)

                            region_of_interest_after = data_a[pot_start:region_stop]
                            #print("region_of_interest_after", region_of_interest_after)

                            if region_of_interest_after.size == 0:
                                mean_after = data_a[pot_start]
                            else:
                                mean_after = np.mean(region_of_interest_after)
                            #print("mean_after", mean_after)

                            if mean_before > mean_after + self.safty_factor:
                                gain_intervals[-1][1] = prev_stop

                                # prevent an infinite loop
                                if last_iteration_borders == [prev_stop, pot_start]:
                                    if iteration_check >= 10:
                                        self.result["error_code"][self.current_idx] = 4
                                        raise InfiniteLoopNumberError("Breaking infinite loop")
                                    iteration_check += 1
                                else:
                                    last_iteration_borders = [prev_stop, pot_start]
                                    iteration_check = 0

                                i += near_matches_after[-1]
                                set_stop_flag = False
                                continue
                    else:
                        if mean_before > mean_after + self.safty_factor:
                            gain_intervals[-1][1] = prev_stop

                            # prevent an infinite loop
                            if last_iteration_borders == [prev_stop, pot_start]:
                                if iteration_check >= 10:
                                    self.result["error_code"][self.current_idx] = 4
                                    raise InfiniteLoopNumberError("Breaking infinite loop")
                                iteration_check += 1
                            else:
                                last_iteration_borders = [prev_stop, pot_start]
                                iteration_check = 0

                            if near_matches_after.size != 0:
                                # do not add +1 here (like it is done if the
                                # interval is closed) because this will result
                                # in one jump in the idx to many
                                i += near_matches_after[-1]
                                set_stop_flag = False

                            # do not increase i
                            continue

                else:
                    if near_matches_after.size != 0:
                        # because near_matches_after starts with 0 -> infinite loop without the +1
                        i += 1 + near_matches_after[-1]
                        set_stop_flag = False
                        continue

                    elif mean_before > mean_after + self.safty_factor:
                        if not set_stop_flag:
                            set_stop_flag = True

                        gain_intervals.append([pot_start, 0])
                i += 1

        if gain_intervals[-1][1] == 0:
            gain_intervals[-1][1] = self.diff.size + 1
        else:
            gain_intervals.append([pot_start, self.diff.size + 1])
        #print("found gain intervals", gain_intervals)
        #print("len gain intervals", len(gain_intervals))

        if len(gain_intervals) > 3:
            self.result["error_code"][self.current_idx] = 2
            raise GainStageNumberError("Too many gain stages found: {}"
                                     .format(gain_intervals))

        if len(gain_intervals) < 3:
            self.result["error_code"][self.current_idx] = 3
            raise GainStageNumberError("Not enough gain stages found: {}"
                                     .format(gain_intervals))

        # store current_idx dependent results in the final matrices
        try:
            self.result["collection"]["diff_changes_idx"][self.current_idx] = (
                self.diff_changes_idx)
        except ValueError:
            # it is not fixed how many diff_changes indices are found
            # but the size of np.arrays has to be fixed
            l = len(self.diff_changes_idx)
            f_diff_changes_idx = (
                self.result["collection"]["diff_changes_idx"][self.current_idx])

            # fill up with found ones, rest of the array stays NAN
            if l < self.n_diff_changes_stored:
                f_diff_changes_idx[:l] = self.diff_changes_idx
            # more regions found than array can hold, only store the first part
            else:
                f_diff_changes_idx = self.diff_changes_idx[:self.n_diff_changes_stored]

        self.result["collection"]["len_diff_changes_idx"][self.current_idx] = (
            len(self.diff_changes_idx))

        self.result["intervals"]["gain_stages"][self.gain_idx["high"]] = gain_intervals[0]
        self.result["intervals"]["gain_stages"][self.gain_idx["medium"]] = gain_intervals[1]
        self.result["intervals"]["gain_stages"][self.gain_idx["low"]] = gain_intervals[2]

    def fit_gain(self, gain, interval, cut_off_ends):

        # the interval is splitted into sub-intervals
        intervals = self.split_interval(interval, self.n_intervals[gain])

        for i in np.arange(len(intervals)):
            self.fit_data(gain, intervals[i], i, cut_off_ends=cut_off_ends)

        return intervals

    def split_interval(self, interval, nsplits):
        """
        split given interval into equaly sized sub-intervals
        """
        #print("nsplits", nsplits)
        #print("interval={}".format(interval))

        step = int((interval[-1] - interval[0]) / nsplits)
        #print("step", step)

        if step == 0:
            print("current_idx", self.current_idx)
            print("nsplits", nsplits)
            print("interval={}".format(interval))
            raise IntervalSplitError("Interval has not enough points to split")

        splitted_intervals = [
            [interval[0] + i * step, interval[0] + (i + 1) * step]
                for i in np.arange(nsplits)]

        # due to rounding or int convertion the last calculated interval ends
        # before the end of the given interval
        splitted_intervals[-1][1] = interval[1]

        #print("splitted_intervals={}".format(splitted_intervals))

        return splitted_intervals

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
        self.data_to_fit[gain][interval_idx] = self.analog[self.current_idx[0], self.current_idx[1],
                                                           self.current_idx[2],
                                                           lower_border:upper_border]

        self.x_values[gain][interval_idx] = np.arange(lower_border, upper_border)
        #TODO check that size of inteval is not to small, i.e. < 3

        number_of_points = len(self.x_values[gain][interval_idx])

        # scaling
        self.scale_x_interval(gain, interval_idx)

        # .T means transposed
        self.coefficient_matrix = np.vstack([self.x_values[gain][interval_idx],
                                             np.ones(number_of_points)]).T

        # fit the data
        # reason to use numpy lstsq:
        # https://stackoverflow.com/questions/29372559/what-is-the-difference-between-numpy-linalg-lstsq-and-scipy-linalg-lstsq
        #lstsq returns: Least-squares solution (i.e. slope and offset), residuals, rank, singular values
        array_idx = self.current_idx + (interval_idx,)
        res = None
        try:
            res = np.linalg.lstsq(self.coefficient_matrix, self.data_to_fit[gain][interval_idx])
            self.result["slope"]["individual"][gain][array_idx] = res[0][0]
            self.result["offset"]["individual"][gain][array_idx] = res[0][1]
            self.result["residuals"]["individual"][gain][array_idx] = res[1]
            self.result["average_residual"]["individual"][gain][array_idx] = (
                np.sqrt(res[1] / number_of_points))
            #print("average_residual", self.result["average_residual"]["individual"][gain][array_idx])
        except:
            if res is None:
                print("interval\n{}".format(interval))
                print("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
                print("self.data_to_fit[{}][{}]\n{}"
                      .format(gain, interval_idx, self.data_to_fit[gain][interval_idx]))

                raise

            if res[0].size != 2:
                raise FitError("Failed to calculate slope and offset")
            elif res[1].size != 1:
                raise FitError("Failed to calculate residual")
            else:
                print("interval\n{}".format(interval))
                print("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
                print("self.data_to_fit[{}][{}]\n{}"
                      .format(gain, interval_idx, self.data_to_fit[gain][interval_idx]))
                print("res", res)

                raise

        #print("found slope: {}".format(self.result["slope"]["individual"][gain]))
        #print("found offset: {}".format(self.result["offset"]["individual"][gain]))

    def calc_gain_median(self):
        for gain in ["high", "medium", "low"]:
            idx = (self.current_idx[0],
                   self.current_idx[1],
                   self.current_idx[2],
                   slice(self.result["intervals"]["gain_stages"][self.gain_idx[gain]][0],
                         self.result["intervals"]["gain_stages"][self.gain_idx[gain]][1]))

            self.result["medians"][self.gain_idx[gain]] = np.median(self.digital[idx])

    def calc_thresholds(self):
        # threshold between high and medium
        self.result["thresholds"][(0, ) + self.current_idx] = np.mean([
                self.result["medians"][self.gain_idx["high"]],
                self.result["medians"][self.gain_idx["medium"]]])

        # threshold between medium and low
        self.result["thresholds"][(1, ) + self.current_idx] = np.mean([
                self.result["medians"][self.gain_idx["medium"]],
                self.result["medians"][self.gain_idx["low"]]])

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

            for key in self.result:
                if type(self.result[key]) != dict:
                    save_file.create_dataset("/{}".format(key), data=self.result[key])
                else:
                    for subkey in self.result[key]:
                        if type(self.result[key][subkey]) != dict:
                            save_file.create_dataset("/{}/{}".format(key, subkey),
                                                     data=self.result[key][subkey])
                        else:
                            for gain in ["high", "medium", "low"]:
                                save_file.create_dataset("/{}/{}/{}".format(key, subkey, gain),
                                                         data=self.result[key][subkey][gain])

            save_file.flush()
            print("took time: {}".format(time.time() - t))
        finally:
            save_file.close()

    def create_plots(self):
        idx = self.current_idx + (slice(None),)

        plot_file_prefix = "{}_[{}, {}]_{}".format(self.plot_file_prefix,
                                                   self.current_idx[0],
                                                   self.current_idx[1],
                                                   str(self.current_idx[2]).zfill(3))
        plot_title_prefix = "{}_[{}, {}]_{}".format(self.plot_title_prefix,
                                                    self.current_idx[0],
                                                    self.current_idx[1],
                                                    str(self.current_idx[2]).zfill(3))

        if type(self.create_plot_type) == list:

            if "all" in self.create_plot_type:
                generate_all_plots(self.current_idx,
                               self.scaled_x_values,
                               self.analog[idx],
                               self.digital[idx],
                               self.x_values,
                               self.data_to_fit,
                               self.result["slope"]["individual"],
                               self.result["offset"]["individual"],
                               self.n_intervals,
                               self.fit_cutoff_left,
                               self.fit_cutoff_right,
                               plot_title_prefix,
                               plot_file_prefix,
                               self.plot_ending)

            else:
                if "data" in self.create_plot_type:
                    plot_title = "{} data".format(plot_title_prefix)
                    plot_name = "{}_data{}".format(plot_file_prefix,
                                                   self.plot_ending)

                    generate_data_plot(self.current_idx,
                                       self.scaled_x_values,
                                       self.analog[idx],
                                       self.digital[idx],
                                       plot_title,
                                       plot_name)

                if "fit" in self.create_plot_type:

                    for gain in ["high", "medium", "low"]:
                        plot_title = "{} fit {}".format(plot_title_prefix, gain)
                        plot_name = "{}_fit_{}{}".format(plot_file_prefix, gain,
                                                         self.plot_ending)

                        generate_fit_plot(self.current_idx,
                                          self.x_values[gain],
                                          self.data_to_fit[gain],
                                          self.result["slope"]["individual"][gain],
                                          self.result["offset"]["individual"][gain],
                                          self.n_intervals[gain],
                                          plot_title,
                                          plot_name)

                if "combined" in self.create_plot_type:
                    plot_title = "{} combined".format(plot_title_prefix)
                    plot_name = "{}_combined{}".format(plot_file_prefix,
                                                       self.plot_ending)
                    generate_combined_plot(self.current_idx,
                                           self.scaled_x_values,
                                           self.analog[idx],
                                           self.digital[idx],
                                           self.fit_cutoff_left,
                                           self.fit_cutoff_right,
                                           self.x_values,
                                           self.result["slope"]["individual"],
                                           self.result["offset"]["individual"],
                                           plot_title, plot_name)
        else:
            generate_all_plots(self.current_idx,
                               self.scaled_x_values,
                               self.analog[idx],
                               self.digital[idx],
                               self.x_values,
                               self.data_to_fit,
                               self.result["slope"]["individual"],
                               self.result["offset"]["individual"],
                               self.n_intervals,
                               self.fit_cutoff_left,
                               self.fit_cutoff_right,
                               plot_title_prefix,
                               plot_file_prefix,
                               self.plot_ending)

if __name__ == "__main__":

    base_dir = "/gpfs/cfel/fsds/labs/agipd/calibration/processed/"

    asic = 10
    #asic = 2
    #asic = 1
    #asic = 11
    module = "M314"
    temperature = "temperature_m15C"
    current = "itestc150"

    input_fname = os.path.join(base_dir, module, temperature, "drscs", current, "gather",
                              "{}_drscs_{}_asic{}.h5".format(module, current, str(asic).zfill(2)))
    output_fname = os.path.join(base_dir, module, temperature, "drscs", current, "process",
                               "{}_drscs_{}_asic{}_processed.h5".format(module, current, str(asic).zfill(2)))
    plot_dir = os.path.join(base_dir, module, temperature, "drscs", "plots", current, "manu_test")
    #plot_dir = os.path.join(base_dir, module, temperature, "drscs", "plots", current, "asic_{}".format(asic))
    plot_prefix = "{}_{}".format(module, current)

    #pixel_v_list = np.arange(64)
    #pixel_u_list = np.arange(64)
    #mem_cell_list = np.arange(352)

    pixel_v_list = np.array([48])
    pixel_u_list = np.array([59])
    mem_cell_list = np.arange(0,1)

    #output_fname = False
    #create_plots can be set to False, "data", "fit", "combined" or "all"
    create_plots = False#["combined"]
    create_error_plots = False #True
    #create_plots=["data", "combined"]

    cal = ProcessDrscs(asic, input_fname, output_fname=output_fname,
                       plot_prefix=plot_prefix, plot_dir=plot_dir,
                       create_plots=create_plots)

    print("\nRun processing")
    t = time.time()
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list,
            create_error_plots=create_error_plots)
    print("Processing took time: {}".format(time.time() - t))
