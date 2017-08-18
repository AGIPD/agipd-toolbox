from __future__ import print_function

from multiprocessing import Pool, TimeoutError
import numpy as np
import h5py
import os
import sys
import time
import traceback
from characterization.plotting import generate_data_plot, generate_fit_plot, generate_combined_plot, generate_all_plots
from helpers import create_dir, check_file_exists, setup_logging
import matplotlib.pyplot as plt
import copy


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
        "residual": {
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
            "region_range_in_percent": None,
            "region_range": None,
            "safety_factor" : None,
            "diff_changes_idx": None,
            "len_diff_changes_idx": None,
            "saturation_threshold": None,
            "saturation_threshold_diff": None,
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
    for key in ["slope", "offset", "residual", "average_residual"]:
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
    result["intervals"]["saturation"] = init_with_nan(shape_tmp + (2,), np.int)

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
                 analog=None, digital=None, safety_factor=1000,
                 plot_prefix=None, plot_dir=None, create_plots=False,
                 log_level="info"):

        self.log = setup_logging("ProcessDrscs", log_level)

        if log_level.lower() == "debug":
            self.use_debug = True
        else:
            self.use_debug = False

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
        self.frame_loss_analog_path = "/entry/instrument/detector/collection/frame_loss_analog"
        self.frame_loss_digital_path = "/entry/instrument/detector/collection/frame_loss_digital"

        self.create_plot_type = create_plots
        self.plot_dir = plot_dir

        self.percent = 10

        self.safety_factor = safety_factor
        self.n_diff_changes_stored = 10
        self.saturation_threshold = 14000
        self.saturation_threshold_diff = 30,

        self.diff_threshold = -100
        self.region_range_in_percent = 2
        self.region_range = 10

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

        self.in_idx = None
        self.out_idx = None

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
        # 5    Failed to calculate slope and offset
        # 6    Not enough point to fit {} gain
        #
        # warning_code:
        # 1    Failed to calculate residual
        ###


        if plot_prefix:
            self.log.info("plot_prefix {}".format(plot_prefix))
            self.plot_file_prefix = "{}_asic{}".format(plot_prefix, str(asic).zfill(2))
            if plot_dir is not None:
                self.plot_file_prefix = os.path.join(self.plot_dir, self.plot_file_prefix)

            self.plot_title_prefix = self.plot_file_prefix.rsplit("/", 1)[1]
            self.log.debug("plot_file_prefix {}".format(self.plot_file_prefix))
            self.log.info("plot_title_prefix {}".format(self.plot_title_prefix))

            self.plot_ending = ".png"
        else:
            self.plot_file_prefix = None
            self.plot_title_prefix = None
            self.plot_ending = None

        if self.output_fname is not None:
            check_file_exists(self.output_fname)

    def run(self, pixel_v_list, pixel_u_list, mem_cell_list,
            create_error_plots=False):

        if len(pixel_v_list) == 0 or len(pixel_u_list) == 0:
            self.log.error("pixel_v_list: {}".format(pixel_v_list))
            self.log.error("pixel_u_list: {}".format(pixel_u_list))
            raise Exception("No proper pixel specified")
        if len(mem_cell_list) == 0:
            self.log.error("mem_cell_list: {}".format(mem_cell_list))
            raise Exception("No proper memory cell specified")

        if create_error_plots and self.plot_title_prefix is None:
            raise Exception("Plotting was not defined on initiation. Quitting.")

        self.pixel_v_list = pixel_v_list
        self.pixel_u_list = pixel_u_list
        self.mem_cell_list = mem_cell_list

        self.check_lists_continous()

        if self.input_fname is not None:
            self.log.info("Load data")
            t = time.time()
            self.load_data()
            self.log.info("took time: {}".format(time.time() - t))

        self.scale_full_x_axis()

        # initiate
        self.result = initiate_result(self.pixel_v_list, self.pixel_u_list,
                                      self.mem_cell_list, self.n_gain_stages,
                                      self.n_intervals, self.n_diff_changes_stored)

        self.result["collection"]["diff_threshold"] = self.diff_threshold
        self.result["collection"]["region_range_in_percent"] = self.region_range_in_percent
        self.result["collection"]["region_range"] = self.region_range
        self.result["collection"]["safety_factor"] = self.safety_factor
        self.result["collection"]["n_diff_changes_stored"] = self.n_diff_changes_stored
        self.result["collection"]["scaling_point"] = self.scaling_point
        self.result["collection"]["scaling_factor"] = self.scaling_factor
        self.result["collection"]["n_gain_stages"] = self.n_gain_stages
        self.result["collection"]["fit_cutoff_left"] = self.fit_cutoff_left
        self.result["collection"]["fit_cutoff_right"] = self.fit_cutoff_right
        self.result["collection"]["saturation_threshold"] = self.saturation_threshold
        self.result["collection"]["saturation_threshold_diff"] = self.saturation_threshold_diff

        for pixel_v in self.pixel_v_list:
            #self.log.debug("start processing row {}".format(pixel_v))
            for pixel_u in self.pixel_u_list:
                #t = time.time()
                for mem_cell in self.mem_cell_list:
                    try:
                        # location in the input data
                        self.in_idx = (pixel_v - self.pixel_v_list[0],
                                       pixel_u - self.pixel_u_list[0],
                                       mem_cell - self.mem_cell_list[0])
                        # location where to write the data to
                        self.out_idx = (pixel_v, pixel_u, mem_cell)
                        self.gain_idx = {
                            "high": (0, ) + self.out_idx,
                            "medium": (1, ) + self.out_idx,
                            "low": (2, ) + self.out_idx
                        }

                        self.process_data_point(self.out_idx)

                        for gain in self.gain_idx:
                            l = self.fit_cutoff_left
                            r = self.fit_cutoff_right
                            if len(self.result["slope"]["individual"][gain][self.out_idx]) >= l + r + 1:
                                for t in ["slope", "offset", "residual", "average_residual"]:
                                    self.result[t]["mean"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain][self.out_idx][l:-r]))
                            else:
                                for t in ["slope", "offset", "residual", "average_residual"]:
                                    self.result[t]["mean"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain][self.out_idx]))

                        if self.result["error_code"][self.out_idx] < 0:
                            self.result["error_code"][self.out_idx] = 0

                    except KeyboardInterrupt:
                        sys.exit(1)
                    except Exception as e:
                        if type(e) == IntervalError:
                            if self.use_debug:
                                self.log.debug("IntervalError")
                            pass
                        elif type(e) == IntervalSplitError:
                            if self.use_debug:
                                self.log.debug("IntervalSplitError")
                            pass
                        elif type(e) == GainStageNumberError:
                            if self.use_debug:
                                self.log.debug("{}: GainStageNumberError".format(self.out_idx))
                                self.log.debug("found number of diff_changes_idx={}"
                                               .format(len(self.diff_changes_idx)))
                            pass
                        elif type(e) == FitError:
                            if self.use_debug:
                                self.log.debug("FitError")
                            pass
                        else:
                            self.log.error("Failed to run for pixel [{}, {}] and mem_cell {}"
                                  .format(self.out_idx[0], self.out_idx[1], self.out_idx[2]))
                            self.log.error(traceback.format_exc())
                            #TODO self.log.exception ?

                            if self.result["error_code"][self.out_idx] <= 0:
                                self.result["error_code"][self.out_idx] = 1

                        if create_error_plots:
                            try:
                                idx = self.in_idx + (slice(None),)

                                plot_file_prefix = ("{}_[{}, {}]_{}"
                                                    .format(self.plot_file_prefix,
                                                            self.out_idx[0],
                                                            self.out_idx[1],
                                                            str(self.out_idx[2]).zfill(3)))
                                plot_title_prefix = ("{}_[{}, {}]_{}"
                                                     .format(self.plot_title_prefix,
                                                             self.out_idx[0],
                                                             self.out_idx[1],
                                                             str(self.out_idx[2]).zfill(3)))
                                plot_title = "{} data".format(plot_title_prefix)
                                plot_name = "{}_data{}".format(plot_file_prefix,
                                                               self.plot_ending)

                                generate_data_plot(self.out_idx,
                                                   self.scaled_x_values,
                                                   self.analog[idx],
                                                   self.digital[idx],
                                                   plot_title,
                                                   plot_name)
                            except:
                                self.log.error("Failed to generate plot")
                                raise

#                        if type(e) not in [IntervalError, IntervalSplitError, GainStageNumberError]:
#                            raise

                #self.log.debug("Process pixel {} took time: {}".format([pixel_v, pixel_u], time.time() - t))

        if self.output_fname is not None:
            self.log.info("writing data")
            self.write_data()

    def check_lists_continous(self):

        v_list = np.arange(self.pixel_v_list[0], self.pixel_v_list[-1] + 1)
        u_list = np.arange(self.pixel_u_list[0], self.pixel_u_list[-1] + 1)
        mem_cell_list = np.arange(self.mem_cell_list[0], self.mem_cell_list[-1] + 1)

        if not (np.array_equal(self.pixel_v_list, v_list)
                or not np.array_equal(self.pixel_u_list, u_list)
                or not np.array_equal(self.mem_cell_list, mem_cell_list)):
            self.log.error("Input is not a continuous list")
            self.log.info("pixel_v_list {}".format(self.pixel_v_list))
            self.log.info("pixel_u_list {}".format(self.pixel_u_list))
            self.log.info("mem_cell_list {}".format(self.mem_cell_list))
            sys.exit(1)

    def load_data(self):

        try:
            source_file = h5py.File(self.input_fname, "r")
        except:
            self.log.error("Unable to open file {}".format(input_fname))
            raise

        idx = (slice(self.pixel_v_list[0], self.pixel_v_list[-1] + 1),
               slice(self.pixel_u_list[0], self.pixel_u_list[-1] + 1),
               slice(self.mem_cell_list[0], self.mem_cell_list[-1] + 1),
               slice(None))

        idx_frame_loss = (slice(None), idx[2], slice(None))

        try:
            # origin data is written as int16 which results in a integer overflow
            # when handling the scaling
            self.analog = source_file[self.analog_path][idx].astype("int32")
            self.digital = source_file[self.digital_path][idx].astype("int32")

            self.frame_loss_analog = source_file[self.frame_loss_analog_path][idx_frame_loss]
            self.frame_loss_digital = source_file[self.frame_loss_digital_path][idx_frame_loss]
        finally:
            source_file.close()

    def process_data_point(self, out_idx):

        data = self.digital[self.in_idx[0], self.in_idx[1], self.in_idx[2], :]

        self.calc_gain_regions()

        self.result["intervals"]["subintervals"]["high"][self.out_idx] = (
            self.fit_gain("high",
                          self.result["intervals"]["gain_stages"][self.gain_idx["high"]],
                          cut_off_ends=False))

        self.result["intervals"]["subintervals"]["medium"][self.out_idx] = (
            self.fit_gain("medium",
                          self.result["intervals"]["gain_stages"][self.gain_idx["medium"]],
                          cut_off_ends=False))

        self.result["intervals"]["subintervals"]["low"][self.out_idx] = (
            self.fit_gain("low",
                          self.result["intervals"]["gain_stages"][self.gain_idx["low"]],
                          cut_off_ends=False))

        self.calc_gain_median()
        self.calc_thresholds()

        if self.create_plot_type:
            self.log.info("\nGenerate plots")
            t = time.time()

            self.create_plots()

            self.log.info("took time: {}".format(time.time() - t))

    def calc_gain_regions(self):
        data_a = self.analog[self.in_idx[0], self.in_idx[1], self.in_idx[2], :].astype(np.float)

        # find out if the col was effected by frame loss
        #n_col_sets = self.frame_loss_analog.shape[0]
        #frame_loss = self.frame_loss_analog[self.in_idx[1] % n_col_sets, self.in_idx[2], :]
        #lsot_frames = np.where(frame_loss != 0)

        lost_frames = np.where(data_a == 0)
        #self.log.debug("lost_frames {}".format(lost_frames))
        data_a[lost_frames] = np.nan

#        print(type(data_a))

        missing = np.isnan(data_a)
        # remember the original indices
        origin_idx = np.arange(data_a.size)[~missing]
        data_a = data_a[~missing]

        # calculates the difference between neighboring elements
        self.diff = np.diff(data_a)

        # an additional threshold is needed to catch cases like this:
        # safety factor : 450
        # 257 : [9600 9239]
        # 258 : [9239 8873]
        # 259 : [8873 8436]
        # region_of_interest_before (prev_stop: 257):  [9496 9493 9542 9558 9576]
        # region_of_interest_after (pot_start: 258): [9239 8873 8436 8390 8395 8389 8388 8413 8425 8417]
        # region_of_interest_before (prev_stop: 257): [9496 9493 9542 9558 9576]
        # region_of_interest_after (pot_start: 259): [8436 8390 8395 8389 8388 8413 8425 8417 8416 8435]
        # here the safety factor would need needed to be decreased too far

        # other reason: all of the pixels with shadow lines would be passing the tests

#        diff_nan_as_bigger = copy.deepcopy(self.diff)
#        diff_nan_as_bigger[np.isnan(self.diff)] = np.inf
        #self.log.debug("diff_nan_as_bigger {}".format(diff_nan_as_bigger))

#        diff_nan_as_lesser = copy.deepcopy(self.diff)
#        diff_nan_as_lesser[np.isnan(self.diff)] = 0
        #self.log.debug("diff_nan_as_lesser {}".format(diff_nan_as_lesser))

#        self.diff_changes_idx = np.where((diff_nan_as_bigger < self.diff_threshold) |
#                                         (diff_nan_as_lesser > self.safety_factor))[0]

        self.diff_changes_idx = np.where((self.diff < self.diff_threshold) |
                                         (self.diff > self.safety_factor))[0]

        if self.use_debug:
            for i in self.diff_changes_idx:
                self.log.debug("{} : {}".format(i, data_a[i:i+2]))

        gain_intervals = [[0, 0]]

        i = 0
        prev_stop = 0
        prev_stop_idx = 0
        pot_start = 0
        set_stop_flag = True
        iteration_check = 0
        last_iteration_borders = []
        region_range_before = 0
        while i < len(self.diff_changes_idx):

            if set_stop_flag:
                prev_stop = self.diff_changes_idx[i]
                prev_stop_idx = i
            # exclude the found point
            pot_start = self.diff_changes_idx[i] + 1

            range_len_tmp = int(np.ceil((prev_stop - gain_intervals[-1][0]) * self.region_range_in_percent / 100))
            if range_len_tmp != 0:
                region_range_before = range_len_tmp
            # the region before would be empty
            else:
                i += 1
                continue

            # determine the region before the potention gain stage change
            start_before = prev_stop - region_range_before
            if start_before < 0:
                start_before = 0

            region_of_interest_before = data_a[start_before:prev_stop]

            # determine the region after the potention gain stage change
            stop_after = pot_start + self.region_range
            if stop_after > self.diff.size:
                stop_after = self.diff.size

            region_of_interest_after = data_a[pot_start:stop_after]

            # check if the following idx is contained in region_of_interest_before
            near_matches_before = np.where(start_before < self.diff_changes_idx[:prev_stop_idx])
            # np.where returns a tuple (array, type)
            near_matches_before = near_matches_before[0]

            # check if the following idx is contained in region_of_interest_after
            near_matches_after = np.where(self.diff_changes_idx[i + 1:] < stop_after)
            # np.where returns a tuple (array, type)
            near_matches_after = near_matches_after[0]
            if region_of_interest_before.size == 0:
                mean_before = data_a[prev_stop]
            elif np.all(np.isnan(region_of_interest_before)):
                region_of_interest_before = np.array([])
                mean_before = np.nan
            else:
                mean_before = np.nanmean(region_of_interest_before)

            if region_of_interest_after.size == 0:
                mean_after = data_a[pot_start]
            elif np.all(np.isnan(region_of_interest_after)):
                region_of_interest_after = np.array([])
                mean_after = np.nan
            else:
                mean_after = np.nanmean(region_of_interest_after)

            if self.use_debug:
                self.log.debug("\n")
                self.log.debug("gain intervals {}".format(gain_intervals))
                self.log.debug("prev_stop: {}".format(prev_stop))
                self.log.debug("pot_start: {}".format(pot_start))

                self.log.debug("region_range_before: {}".format(region_range_before))
                self.log.debug("region_of_interest_before {}".format(region_of_interest_before))
                self.log.debug("region_of_interest_after: {}".format(region_of_interest_after))

                self.log.debug("near match before {} {}"
                               .format(near_matches_before,
                                       self.diff_changes_idx[:prev_stop_idx][near_matches_before]))
                self.log.debug("near match after {} {}"
                               .format(near_matches_after,
                                       self.diff_changes_idx[i + 1:][near_matches_after]))

                self.log.debug("mean_before {}".format(mean_before))
                self.log.debug("mean_after {}".format(mean_after))

            if near_matches_before.size == 0 and near_matches_after.size == 0:

                # if the slope is too high the mean is falsified by this
                if (region_of_interest_before.size != 0 and \
                        np.max(region_of_interest_before)
                        - np.min(region_of_interest_before) > self.safety_factor * 2):
                    # cut the region of interest into half
                    new_region = region_of_interest_before[region_of_interest_before.size // 2:]
                    mean_before = np.nanmean(new_region)

                    if self.use_debug:
                        self.log.debug("mean_before after cut down of region of interest: {}"
                                       .format(mean_before))

                if mean_before > mean_after + self.safety_factor:
                    # a stage change was found
                    gain_intervals[-1][1] = prev_stop
                    gain_intervals.append([pot_start, 0])

                i += 1
                set_stop_flag = True

            else:
                if gain_intervals[-1][1] == 0:
                    if self.use_debug:
                        self.log.debug("gain intervals last entry == 0")

                    if near_matches_before.size != 0:
                        region_start = self.diff_changes_idx[:prev_stop_idx][near_matches_before[-1]] + 1

                        region_of_interest_before = data_a[region_start:prev_stop]

                        if region_of_interest_before.size == 0:
                            mean_before = data_a[prev_stop]
                        elif np.all(np.isnan(region_of_interest_before)):
                            mean_before = np.nan
                        else:
                            mean_before = np.nanmean(region_of_interest_before)

                        if self.use_debug:
                            self.log.debug("near_matches_before is not empty")
                            self.log.debug("region_start {}".format(region_start))
                            self.log.debug("region_of_interest_before {}".format(region_of_interest_before))
                            self.log.debug("mean_before {}".format(mean_before))

                    if near_matches_before.size == 0:
                        if self.use_debug:
                            self.log.debug("near_matches_before.size == 0")

                        if mean_before > mean_after + self.safety_factor:
                            if self.use_debug:
                                self.log.debug("mean check")

                            gain_intervals[-1][1] = prev_stop

                            # prevent an infinite loop
                            if last_iteration_borders == [prev_stop, pot_start]:
                                if iteration_check >= 10:
                                    self.result["error_code"][self.out_idx] = 4
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
                            region_start = self.diff_changes_idx[i + 1:][near_matches_after[-1]]
                            region_of_interest_after = data_a[region_start:stop_after]

                            if region_of_interest_after.size == 0:
                                mean_after = data_a[pot_start]
                            elif np.all(np.isnan(region_of_interest_after)):
                                mean_after = np.nan
                            else:
                                mean_after = np.nanmean(region_of_interest_after)

                            if self.use_debug:
                                self.log.debug("near_matches_after is not empty")
                                self.log.debug("region_start {}".format(region_start))
                                self.log.debug("region_of_interest_after {}"
                                               .format(region_of_interest_after))
                                self.log.debug("mean_after {}".format(mean_after))

                            if mean_before > mean_after + self.safety_factor:
                                gain_intervals[-1][1] = prev_stop

                                # prevent an infinite loop
                                if last_iteration_borders == [prev_stop, pot_start]:
                                    if iteration_check >= 10:
                                        self.result["error_code"][self.out_idx] = 4
                                        raise InfiniteLoopNumberError("Breaking infinite loop")
                                    iteration_check += 1
                                else:
                                    last_iteration_borders = [prev_stop, pot_start]
                                    iteration_check = 0

                                i += near_matches_after[-1]
                                set_stop_flag = False
                                continue
                            # if there is an outlier right before an jump but the
                            # slope is so steep that not considering the region
                            # between the outlier and the jump would falsify the
                            # jump detection
                            elif mean_before + self.safety_factor < mean_after:
                                if self.use_debug:
                                    self.log.debug("mean before is much bigger than mean after")

                                set_stop_flag = True
                            else:
                                set_stop_flag = False
                    else:
                        if mean_before > mean_after + self.safety_factor:
                            gain_intervals[-1][1] = prev_stop

                            # prevent an infinite loop
                            if last_iteration_borders == [prev_stop, pot_start]:
                                if iteration_check >= 10:
                                    self.result["error_code"][self.out_idx] = 4
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

                    elif mean_before > mean_after + self.safety_factor:
                        if not set_stop_flag:
                            set_stop_flag = True

                        gain_intervals.append([pot_start, 0])
                i += 1

        if gain_intervals[-1][1] == 0:
            gain_intervals[-1][1] = self.diff.size + 1
        else:
            gain_intervals.append([pot_start, self.diff.size + 1])

        # map the found indices back to the origin data
        mapped_gain_intervals = []
        for idx in np.arange(len(gain_intervals) - 1):
            i = gain_intervals[idx]
            mapped_gain_intervals.append([origin_idx[i[0]], origin_idx[i[1]]])
        mapped_gain_intervals.append([origin_idx[gain_intervals[-1][0]], origin_idx[-1]])

        gain_intervals = mapped_gain_intervals

        if self.use_debug:
            self.log.debug("{}, found gain intervals {}".format(self.out_idx, gain_intervals))
            self.log.debug("len gain intervals {}".format(len(gain_intervals)))

        if len(gain_intervals) > 3:
            self.result["error_code"][self.out_idx] = 2
            raise GainStageNumberError("Too many gain stages found: {}"
                                     .format(gain_intervals))

        if len(gain_intervals) < 3:
            self.result["error_code"][self.out_idx] = 3
            raise GainStageNumberError("Not enough gain stages found: {}"
                                     .format(gain_intervals))

        # store out_idx dependent results in the final matrices
        try:
            self.result["collection"]["diff_changes_idx"][self.out_idx] = (
                self.diff_changes_idx)
        except ValueError:
            # it is not fixed how many diff_changes indices are found
            # but the size of np.arrays has to be fixed
            l = len(self.diff_changes_idx)
            f_diff_changes_idx = (
                self.result["collection"]["diff_changes_idx"][self.out_idx])

            # fill up with found ones, rest of the array stays NAN
            if l < self.n_diff_changes_stored:
                f_diff_changes_idx[:l] = self.diff_changes_idx
            # more regions found than array can hold, only store the first part
            else:
                f_diff_changes_idx = self.diff_changes_idx[:self.n_diff_changes_stored]

        self.result["collection"]["len_diff_changes_idx"][self.out_idx] = (
            len(self.diff_changes_idx))

        self.result["intervals"]["gain_stages"][self.gain_idx["high"]] = gain_intervals[0]
        self.result["intervals"]["gain_stages"][self.gain_idx["medium"]] = gain_intervals[1]

        new_gain_stage = self.detect_saturation(gain_intervals[2])
        #self.log.debug("new_gain_stage: {}".format(new_gain_stage))
        self.result["intervals"]["gain_stages"][self.gain_idx["low"]] = new_gain_stage


    def detect_saturation(self, interval):

        idx = (self.in_idx[0],
               self.in_idx[1],
               self.in_idx[2],
               slice(interval[0], interval[1]))

        sat_index = np.where(self.analog[idx] >= self.saturation_threshold)[0]

        # check if the result of where was empty
        if sat_index.size == 0:
            sat_index = interval[1]
        else:
            sat_index = interval[0] + sat_index[0]
        #self.log.debug("sat_indices {}".format(sat_index))
        #self.log.debug("value {}".format(self.analog[self.in_idx[0], self.in_idx[1], self.in_idx[2], sat_index-5:sat_index+6])

        new_interval = [interval[0], sat_index]

        # second iteration for saturation detection
        new_gain_interval, new_sat_interval = self.detect_saturation_with_diff(new_interval)

        #self.log.debug("new_gain_interval {}".format(new_gain_interval))
        #self.log.debug("new_sat_interval {}".format(new_sat_interval))

        self.result["intervals"]["saturation"][self.out_idx] = [new_sat_interval[0], interval[1]]

        return new_gain_interval
        #return [interval[0], sat_index]


    def detect_saturation_with_diff(self, interval):
        diff_det_interval = self.diff[interval[0]:interval[1]]
        #self.log.debug("diff_det_interval {}".format(diff_det_interval))
        #self.log.debug("saturation_threshold {}".format(self.saturation_threshold_diff))

        sat_indices = np.where(np.absolute(diff_det_interval) < self.saturation_threshold_diff)[0]
        #self.log.debug("sat_indices {}".format(sat_indices))

        j = sat_indices[-1]
        for i in sat_indices[::-1]:
            if i == j:
                j -= 1
            else:
                #self.log.debug("not true for {} {}".format(i, diff_det_interval[i-1:i+2]))
                break
        #self.log.debug("i {}".format(i))
        #self.log.debug("j {}".format(j))

        saturation_interval = [interval[0] + i, interval[1]]
        new_gain_interval = [interval[0] + sat_indices[0], interval[0] + i]

        return new_gain_interval, saturation_interval


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
        #self.log.debug("nsplits {}".format(nsplits))
        #self.log("interval={}".format(interval))

        step = int((interval[-1] - interval[0]) / nsplits)
        #self.log.debug("step {}".format(step))

        if step == 0:
            self.log.debug("out_idx {}".format(self.out_idx))
            self.log.debug("nsplits {}".format(nsplits))
            self.log.debug("interval={}".format(interval))
            raise IntervalSplitError("Interval has not enough points to split")

        splitted_intervals = [
            [interval[0] + i * step, interval[0] + (i + 1) * step]
                for i in np.arange(nsplits)]

        # due to rounding or int convertion the last calculated interval ends
        # before the end of the given interval
        splitted_intervals[-1][1] = interval[1]

        #self.log.debug("splitted_intervals={}".format(splitted_intervals))

        return splitted_intervals

    def fit_data(self, gain, interval, interval_idx, cut_off_ends=False):

        # find the inner points (cut off the top and bottom part
        if cut_off_ends:
            tmp = np.arange(interval[0], interval[1])
            # 100.0 is needed because else it casts it as ints, i.e. 10/100=>0
            lower_border = int(tmp[0] + len(tmp) * self.percent/100.0)
            upper_border = int(tmp[0] + len(tmp) * (1 - self.percent/100.0))
            #self.log.debug("lower_border = {}".format(lower_border))
            #self.log.debug("upper_border = {}".format(upper_border))
        else:
            lower_border = interval[0]
            upper_border = interval[1]

        # transform the problem y = mx + c
        # into the form y = Ap, where A = [[x 1]] and p = [[m], [c]]
        # meaning  data_to_fit = A * [slope, offset]
        y = self.analog[self.in_idx[0], self.in_idx[1], self.in_idx[2],
                        lower_border:upper_border].astype(np.float)

        self.x_values[gain][interval_idx] = np.arange(lower_border, upper_border)
        #TODO check that size of inteval is not to small, i.e. < 3

        # scaling
        self.scale_x_interval(gain, interval_idx)

        # find out if the col was effected by frame loss
        #n_col_sets = self.frame_loss_analog.shape[0]
        #frame_loss = self.frame_loss_analog[self.in_idx[1] % n_col_sets, self.in_idx[2],
        #                                    lower_border:upper_border]
        #lost_frames = np.array(np.where(frame_loss != 0))

        lost_frames = np.where(y == 0)
        y[lost_frames] = np.NAN

        # remove the ones with frameloss
        missing = np.isnan(y)
        self.data_to_fit[gain][interval_idx] = y[~missing]
        x = self.x_values[gain][interval_idx][~missing]

        number_of_points = len(x)

        # .T means transposed
        A = np.vstack([x, np.ones(number_of_points)]).T

        if A.size <= 1 or self.data_to_fit[gain][interval_idx].size <= 1:
            self.result["error_code"][self.out_idx] = 6
            msg = ("{}: Not enough point to fit {} gain (number of point available: {})"
                   .format(self.out_idx, gain, self.data_to_fit[gain][interval_idx].size))

            if self.use_debug:
                self.log.debug(msg)

            raise FitError(msg)

        # fit the data
        # reason to use numpy lstsq:
        # https://stackoverflow.com/questions/29372559/what-is-the-difference-between-numpy-linalg-lstsq-and-scipy-linalg-lstsq
        #lstsq returns: Least-squares solution (i.e. slope and offset), residuals, rank, singular values
        array_idx = self.out_idx + (interval_idx,)
        res = None
        try:
            res = np.linalg.lstsq(A, self.data_to_fit[gain][interval_idx])

            self.result["slope"]["individual"][gain][array_idx] = res[0][0]
            self.result["offset"]["individual"][gain][array_idx] = res[0][1]
            self.result["residual"]["individual"][gain][array_idx] = res[1]
            self.result["average_residual"]["individual"][gain][array_idx] = (
                np.sqrt(res[1] / number_of_points))

            #self.log.debug("average_residual {}".format(self.result["average_residual"]["individual"][gain][array_idx]))
        except:
            if res is None:
                self.log.debug("interval {}".format(interval))
                self.log.debug("A {}".format(A))
                self.log.debug("self.data_to_fit[{}][{}] {}"
                               .format(gain, interval_idx,
                                       self.data_to_fit[gain][interval_idx]))

                raise

            if res[0].size != 2:
                self.result["error_code"][self.out_idx] = 5
                msg = "Failed to calculate slope and offset"

                if self.use_debug:
                    self.log.debug(msg)

                raise FitError(msg)

            elif res[1].size != 1:
                self.result["warning_code"][self.out_idx] = 1
                msg = "Failed to calculate residual"

                if self.use_debug:
                    self.log.debug("interval {}".format(interval))
                    self.log.debug(res)
                    self.log.debug("self.data_to_fit[{}][{}] {}"
                                   .format(gain, interval_idx,
                                           self.data_to_fit[gain][interval_idx]))
                    self.log.debug(msg)

                raise FitError(msg)
            else:
                self.log.debug("interval\n{}".format(interval))
                self.log.debug("A\n{}".format(A))
                self.log.debug("self.data_to_fit[{}][{}]\n{}"
                               .format(gain, interval_idx,
                                       self.data_to_fit[gain][interval_idx]))
                self.log.debug("res {}".format(res))

                raise

        #self.log.debug("found slope: {}".format(self.result["slope"]["individual"][gain]))
        #self.log.debug("found offset: {}".format(self.result["offset"]["individual"][gain]))

    def calc_gain_median(self):
        for gain in ["high", "medium", "low"]:
            idx = (self.in_idx[0],
                   self.in_idx[1],
                   self.in_idx[2],
                   slice(self.result["intervals"]["gain_stages"][self.gain_idx[gain]][0],
                         self.result["intervals"]["gain_stages"][self.gain_idx[gain]][1]))

            self.result["medians"][self.gain_idx[gain]] = np.median(self.digital[idx])

    def calc_thresholds(self):
        # threshold between high and medium
        self.result["thresholds"][(0, ) + self.out_idx] = np.mean([
                self.result["medians"][self.gain_idx["high"]],
                self.result["medians"][self.gain_idx["medium"]]])

        # threshold between medium and low
        self.result["thresholds"][(1, ) + self.out_idx] = np.mean([
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
            self.log.info("\nStart saving data")
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
                                                         data=self.result[key][subke352n])

            save_file.flush()
            self.log.info("took time: {}".format(time.time() - t))
        finally:
            save_file.close()

    def create_plots(self):
        idx = self.in_idx + (slice(None),)

        plot_file_prefix = "{}_[{}, {}]_{}".format(self.plot_file_prefix,
                                                   self.out_idx[0],
                                                   self.out_idx[1],
                                                   str(self.out_idx[2]).zfill(3))
        plot_title_prefix = "{}_[{}, {}]_{}".format(self.plot_title_prefix,
                                                    self.out_idx[0],
                                                    self.out_idx[1],
                                                    str(self.out_idx[2]).zfill(3))

        if type(self.create_plot_type) == list:

            if "all" in self.create_plot_type:
                generate_all_plots(self.out_idx,
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

                    generate_data_plot(self.out_idx,
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

                        generate_fit_plot(self.out_idx,
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
                    generate_combined_plot(self.out_idx,
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
            generate_all_plots(self.out_idx,
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

    asic = 1
    module = "M215"
    temperature = "temperature_m15C"
    current = "itestc80"
    #safety_factor = 750
    safety_factor = 950

    input_fname = os.path.join(base_dir, module, temperature, "drscs", current, "gather",
                              "{}_drscs_{}_asic{}.h5".format(module, current, str(asic).zfill(2)))
    output_fname = os.path.join(base_dir, module, temperature, "drscs", current, "process",
                               "{}_drscs_{}_asic{}_processed.h5".format(module, current,
                                                                        str(asic).zfill(2)))

    plot_subdir = "pixel_investigation"
    plot_dir = os.path.join(base_dir, module, temperature, "drscs", "plots", current, plot_subdir)

    #plot_dir = os.path.join(base_dir, module, temperature, "drscs", "plots", current, "asic_{}".format(asic))
    plot_prefix = "{}_{}".format(module, current)

    create_dir(plot_dir)

    #pixel_v_list = np.arange(2,3)
    #pixel_u_list = np.arange(38,39)
    #mem_cell_list = np.arange(99,100)

    #pixel_v_list = np.arange(0, 1)
    #pixel_u_list = np.arange(64)
    #mem_cell_list = np.arange(352)

    pixel_v_list = np.array([30])
    pixel_u_list = np.array([2])
    mem_cell_list = np.array([258])
    #mem_cell_list = np.arange(257,258)

    output_fname = False
    #create_plots can be set to False, "data", "fit", "combined" or "all"
    create_plots = ["data", "combined"]
    create_error_plots = True
    #create_plots=["data", "combined"]

    log_level = "info"
    log_level = "debug"

    cal = ProcessDrscs(asic,
                       input_fname,
                       output_fname=output_fname,
                       safety_factor=safety_factor,
                       plot_prefix=plot_prefix,
                       plot_dir=plot_dir,
                       create_plots=create_plots,
                       log_level=log_level)

    print("\nRun processing")
    t = time.time()
    cal.run(pixel_v_list, pixel_u_list, mem_cell_list,
            create_error_plots=create_error_plots)
    print("Processing took time: {}".format(time.time() - t))
