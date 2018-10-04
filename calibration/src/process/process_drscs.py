# (c) Copyright 2017-2018 DESY, FS-DS
#
# This file is part of the FS-DS AGIPD toolbox.
#
# This software is free: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this software.  If not, see <http://www.gnu.org/licenses/>.

"""
@author: Manuela Kuhn <manuela.kuhn@desy.de>
         Jennifer Poehlsen <jennifer.poehlsen@desy.de>
"""

import numpy as np
import time
import h5py
import sys

from process_base import ProcessBase, NotSupported

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

def init_with_nan(shape, dtype):
    # create array
    obj = np.empty(shape, dtype=dtype)
    # intiate with nan
    obj[...] = np.NAN

    return obj

class ProcessDrscs(ProcessBase):
#    def __init__(self, asic, in_fname=False, out_fname=False,
#                 analog=None, digital=None, safety_factor=1000,
#                 plot_prefix=None, plot_dir=None, create_plots=False,
#                 log_level="info"):
    def __init__(self, **kwargs):
        
#        self.fit_interval = None
        self.n_offsets = 3
        super().__init__(**kwargs)
        self.collection: {"run_number": used_run_numbers,
                           "creation_date": str(date.today()),
                           "version": __version__,
                          "diff_threshold": None,
                          "region_range_in_percent": None,
                          "region_range": None,
                          "safety_factor" : None,
                          "diff_changes_idx": None,
                          "len_diff_changes_idx": None,
                          "saturation_threshold": None,
                          "scaling_point": None,
                          "scaling_factor": None,
                          "n_gain_stages": None,
                          "fit_cutoff_left": None,
                          "fit_cutoff_right": None,
                      }

    def initiate(self):

        self.shapes = {
            "high": (self.n_rows,
                     self.n_cols,
                     self.n_memcells,
                     1),

            "low": (self.n_rows,
                    self.n_cols,
                    self.n_memcells,
                    5),

            "mean": (3,
                     self.n_rows,
                     self.n_cols,
                     self.n_memcells),

            "threshold": (2,
                          self.n_rows,
                          self.n_cols,
                          self.n_memcells),

            "code": (self.n_rows,
                     self.n_cols,
                     self.n_memcells),

            "gain_stages": (3,
                            self.n_rows,
                            self.n_cols,
                            self.n_memcells,
                            2),

            "subinterval_high": (self.n_rows,
                                 self.n_cols,
                                 self.n_memcells,
                                 1,
                                 2),

            "subinterval_low": (self.n_rows,
                                self.n_cols,
                                self.n_memcells,
                                5,
                                2)
        }


        self.result = {
            "slope": {
                "mean": {
                    "data": np.empty(self.shapes["mean"]),
                    "path": "slope/mean",
                    "type": np.float32
                },
                "individual": {
                    "high" : {
                        "data": np.empty(self.shapes["high"]),
                        "path": "slope/individual/high",
                        "type": np.float
                    },
                    "medium": {
                        "data": np.empty(self.shapes["high"]),
                        "path": "slope/individual/medium",
                        "type": np.float
                    },
                    "low": {
                        "data": np.empty(self.shapes["low"]),
                        "path": "slope/individual/low",
                        "type": np.float
                    }
                }
            },
            "offset": {
                "mean": {
                    "data": np.empty(self.shapes["mean"]),
                    "path": "offset/mean",
                    "type": np.float32
                },
                "individual": {
                    "high" : {
                        "data": np.empty(self.shapes["high"]),
                        "path": "offset/individual/high",
                        "type": np.float
                    },
                    "medium": {
                        "data": np.empty(self.shapes["high"]),
                        "path": "offset/individual/medium",
                        "type": np.float
                    },
                    "low": {
                        "data": np.empty(self.shapes["low"]),
                        "path": "offset/individual/low",
                        "type": np.float
                    }
                }
            },
            "residual": {
                "mean": {
                    "data": np.empty(self.shapes["mean"]),
                    "path": "residual/mean",
                    "type": np.float32
                },
                "individual": {
                    "high" : {
                        "data": np.empty(self.shapes["high"]),
                        "path": "residual/individual/high",
                        "type": np.float
                    },
                    "medium": {
                        "data": np.empty(self.shapes["high"]),
                        "path": "residual/individual/medium",
                        "type": np.float
                    },
                    "low": {
                        "data": np.empty(self.shapes["low"]),
                        "path": "residual/individual/low",
                        "type": np.float
                    }
                }
            },
            "average_residual": {
                "mean": {
                    "data": np.empty(self.shapes["mean"]),
                    "path": "average_residual/mean",
                    "type": np.float32
                },
                "individual": {
                    "high" : {
                        "data": np.empty(self.shapes["high"]),
                        "path": "average_residual/individual/high",
                        "type": np.float
                    },
                    "medium": {
                        "data": np.empty(self.shapes["high"]),
                        "path": "average_residual/individual/medium",
                        "type": np.float
                    },
                    "low": {
                        "data": np.empty(self.shapes["low"]),
                        "path": "average_residual/individual/low",
                        "type": np.float
                    }
                }
            },
            "intervals": {
                "gain_stages": {
                    "data": np.empty(self.shapes["gain_stages"]),
                    "path": "intervals/gain_stages",
                    "type": np.int64
                },
                "subintervals": {
                    "high" : {
                        "data": np.empty(self.shapes["subinterval_high"]),
                        "path": "intervals/subintervals/high",
                        "type": np.float
                    },
                    "medium": {
                        "data": np.empty(self.shapes["subinterval_high"]),
                        "path": "intervals/subintervals/medium",
                        "type": np.float
                    },
                    "low": {
                        "data": np.empty(self.shapes["subinterval_low"]),
                        "path": "intervals/subintervals/low",
                        "type": np.float
                    }
                }
            },
            "medians": {
                "data": np.empty(self.shapes["mean"]),
                "path": "medians",
                "type": np.float32
            },
            "thresholds": {
                "data": np.empty(self.shapes["threshold"]),
                "path": "thresholds",
                "type": np.float32
            },
            "error_code": {
                "data": np.empty(self.shapes["code"]),
                "path": "error_code",
                "type": np.int16
            },
            "warning_code": {
                "data": np.empty(self.shapes["code"]),
                "path": "warning_code",
                "type": np.int16
            }
        }

        print("n_memcell={}, n_rows={}, n_cols={}".format(self.n_memcells, self.n_rows, self.n_cols))
#        self.log = setup_logging("ProcessDrscs", log_level)
#
#        if log_level.lower() == "debug":
#            self.use_debug = True
#        else:
#            self.use_debug = False

        create_plots = False
        self.create_plot_type = create_plots
        plot_dir = None
        self.plot_dir = plot_dir
        create_error_plots = False
        self.create_error_plots = create_error_plots
        self.use_debug = False

        self.percent = 10

        safety_factor = 950
        self.safety_factor = safety_factor
        self.n_diff_changes_stored = 10
        self.saturation_threshold = 30

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
        #
        # warning_code:
        # 1    Failed to calculate residual
        ###

        plot_prefix = None
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

        #if self.output_fname is not None:
        #    check_file_exists(self.output_fname)

    #def run(self, pixel_v_list, pixel_u_list, mem_cell_list,
    #        create_error_plots=False):
    def calculate(self):

        # if len(pixel_v_list) == 0 or len(pixel_u_list) == 0:
        #     self.log.error("pixel_v_list: {}".format(pixel_v_list))
        #     self.log.error("pixel_u_list: {}".format(pixel_u_list))
        #     raise Exception("No proper pixel specified")
        # if len(mem_cell_list) == 0:
        #     self.log.error("mem_cell_list: {}".format(mem_cell_list))
        #     raise Exception("No proper memory cell specified")

        if self.create_error_plots and self.plot_title_prefix is None:
            raise Exception("Plotting was not defined on initiation. Quitting.")

        # make sure that the data looks like expected
        if (self._row_location > self._col_location
                and self._col_location > self._memcell_location):
            print("found location:\nrow: {}, col: {}, memcell: {}"
                  .format(self._row_location,
                          self._col_location,
                          self._memcell_location))
            print("required order: "
                  "row_location, col_location, memcell_location")
            raise Exception("data layout does not fit algorithm")


        self.pixel_v_list = np.arange(self.n_rows)
        self.pixel_u_list = np.arange(self.n_cols)
        self.mem_cell_list = np.arange(self.n_memcells)

        self.check_lists_continous()

        if self.in_fname is not None:
            #self.log.info("Load data")
            t = time.time()
            self.analog, self.digital = self.load_data(self.in_fname)
            #self.log.info("took time: {}".format(time.time() - t))

        self.scale_full_x_axis()

        self.collection["diff_threshold"] = self.diff_threshold
        self.collection["region_range_in_percent"] = self.region_range_in_percent
        self.collection["region_range"] = self.region_range
        self.collection["safety_factor"] = self.safety_factor
        self.collection["n_diff_changes_stored"] = self.n_diff_changes_stored
        self.collection["scaling_point"] = self.scaling_point
        self.collection["scaling_factor"] = self.scaling_factor
        self.collection["n_gain_stages"] = self.n_gain_stages
        self.collection["fit_cutoff_left"] = self.fit_cutoff_left
        self.collection["fit_cutoff_right"] = self.fit_cutoff_right
        self.collection["saturation_threshold"] = self.saturation_threshold


        # instead of directly accessing the data
        # like analog[slice(*fit_interval), mc, row, col]
        # use a more generic approach to make changes in
        # the data layout easier to fix
        data_slice = [slice(None), slice(None),
                      slice(None), slice(None)]

        print("Start fitting")
        for row in range(self.n_rows):
            print("row {}".format(row))
            data_slice[self._row_location] = row

            for col in range(self.n_cols):
            #for col in range(1):
                data_slice[self._col_location] = col

                for mc in range(self.n_memcells):
                    data_slice[self._memcell_location] = mc

                    # for i in range(self.n_offsets):
                    #     fit_interval = self.fit_interval[i]

                    #     data_slice[self._frame_location] = (
                    #         slice(*fit_interval)
                    #     )

                    try:
                        
                        # location in the input data
                        self.in_idx = (row - self.pixel_v_list[0],
                                       col - self.pixel_u_list[0],
                                       mc - self.mem_cell_list[0])
                        # location where to write the data to
                        self.out_idx = (row, col, mc)
                        self.gain_idx = {
                            "high": (0, ) + self.out_idx,
                            "medium": (1, ) + self.out_idx,
                            "low": (2, ) + self.out_idx
                        }

                        self.process_data_point(self.out_idx)

                        for gain in self.gain_idx:
                            l = self.fit_cutoff_left
                            r = self.fit_cutoff_right
                            if len(self.result["slope"]["individual"][gain]["data"][self.out_idx]) >= l + r + 1:
                                for t in ["slope", "offset", "residual", "average_residual"]:
                                    self.result[t]["mean"]["data"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain]["data"][self.out_idx][l:-r]))
                            else:
                                for t in ["slope", "offset", "residual", "average_residual"]:
                                    self.result[t]["mean"]["data"][self.gain_idx[gain]] = (
                                        np.mean(self.result[t]["individual"][gain]["data"][self.out_idx]))

                        if self.result["error_code"]["data"][self.out_idx] < 0:
                            self.result["error_code"]["data"][self.out_idx] = 0

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
                            #print("{}: GainStageNumberError".format(self.out_idx))
                            #print("found number of diff_changes_idx={}".format(len(self.diff_changes_idx)))
                            pass
                        elif type(e) == FitError:
                            pass
                        else:
                            print("Failed to run for pixel [{}, {}] and mem_cell {}"
                                  .format(self.out_idx[0], self.out_idx[1], self.out_idx[2]))
                            #self.log.error("Failed to run for pixel [{}, {}] and mem_cell {}"
                            #               .format(self.out_idx[0], self.out_idx[1], self.out_idx[2]))
                            #self.log.error(traceback.format_exc())
                            #TODO self.log.exception ?
                            
                            if self.result["error_code"]["data"][self.out_idx] <= 0:
                                self.result["error_code"]["data"][self.out_idx] = 1

                        if self.create_error_plots:
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

        # if self.out_fname is not None:
        #     self.log.info("writing data")
        #     self.write_data()


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

    # def load_data(self):

    #     source_file = h5py.File(self.in_fname, "r")

    #     idx = (slice(self.pixel_v_list[0], self.pixel_v_list[-1] + 1),
    #            slice(self.pixel_u_list[0], self.pixel_u_list[-1] + 1),
    #            slice(self.mem_cell_list[0], self.mem_cell_list[-1] + 1),
    #            slice(None))

    #     # origin data is written as int16 which results in a integer overflow
    #     # when handling the scaling
    #     self.analog = source_file[self.analog_path][idx].astype("int32")
    #     self.digital = source_file[self.digital_path][idx].astype("int32")

    #     source_file.close()

    def process_data_point(self, out_idx):

        data = self.digital[self.in_idx[0], self.in_idx[1], self.in_idx[2], :]

        self.calc_gain_regions()

        self.result["intervals"]["subintervals"]["high"]["data"][self.out_idx] = (
            self.fit_gain("high",
                          self.result["intervals"]["gain_stages"]["data"][self.gain_idx["high"]],
                          cut_off_ends=False))

        self.result["intervals"]["subintervals"]["medium"]["data"][self.out_idx] = (
            self.fit_gain("medium",
                          self.result["intervals"]["gain_stages"]["data"][self.gain_idx["medium"]],
                          cut_off_ends=False))

        self.result["intervals"]["subintervals"]["low"]["data"][self.out_idx] = (
            self.fit_gain("low",
                          self.result["intervals"]["gain_stages"]["data"][self.gain_idx["low"]],
                          cut_off_ends=False))

        self.calc_gain_median()
        self.calc_thresholds()

        if self.create_plot_type:
            self.log.info("\nGenerate plots")
            t = time.time()

            self.create_plots()

            self.log.info("took time: {}".format(time.time() - t))

    def calc_gain_regions(self):
        data_a = self.analog[self.in_idx[0], self.in_idx[1], self.in_idx[2], :]

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
        self.diff_changes_idx = np.where((self.diff < self.diff_threshold) |
                                         (self.diff > self.safety_factor))[0]
#        if self.use_debug:
#            for i in self.diff_changes_idx:
#                self.log.debug("{} : {}".format(i, data_a[i:i+2]))

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
            else:
                mean_before = np.mean(region_of_interest_before)

            if region_of_interest_after.size == 0:
                mean_after = data_a[pot_start]
            else:
                mean_after = np.mean(region_of_interest_after)


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
                    mean_before = np.mean(region_of_interest_before[int(len(region_of_interest_before) / 2):])

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
                        else:
                            mean_before = np.mean(region_of_interest_before)

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
                                    self.result["error_code"]["data"][self.out_idx] = 4
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
                            else:
                                mean_after = np.mean(region_of_interest_after)

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
                                        self.result["error_code"]["data"][self.out_idx] = 4
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
                                    self.result["error_code"]["data"][self.out_idx] = 4
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

        if self.use_debug:
            self.log.debug("{}, found gain intervals {}".format(self.out_idx, gain_intervals))
            self.log.debug("len gain intervals {}".format(len(gain_intervals)))

        if len(gain_intervals) > 3:
            self.result["error_code"]["data"][self.out_idx] = 2
            raise GainStageNumberError("Too many gain stages found: {}"
                                     .format(gain_intervals))

        if len(gain_intervals) < 3:
            self.result["error_code"]["data"][self.out_idx] = 3
            raise GainStageNumberError("Not enough gain stages found: {}"
                                     .format(gain_intervals))

        # store out_idx dependent results in the final matrices
        try:
            self.collection["diff_changes_idx"][self.out_idx] = (
                self.diff_changes_idx)
        except ValueError:
            # it is not fixed how many diff_changes indices are found
            # but the size of np.arrays has to be fixed
            l = len(self.diff_changes_idx)
            f_diff_changes_idx = (
                self.collection["diff_changes_idx"][self.out_idx])

            # fill up with found ones, rest of the array stays NAN
            if l < self.n_diff_changes_stored:
                f_diff_changes_idx[:l] = self.diff_changes_idx
            # more regions found than array can hold, only store the first part
            else:
                f_diff_changes_idx = self.diff_changes_idx[:self.n_diff_changes_stored]

        self.collection["len_diff_changes_idx"][self.out_idx] = (
            len(self.diff_changes_idx))

        self.result["intervals"]["gain_stages"]["data"][self.gain_idx["high"]] = gain_intervals[0]
        self.result["intervals"]["gain_stages"]["data"][self.gain_idx["medium"]] = gain_intervals[1]

        new_gain_stage = self.detect_saturation(gain_intervals[2])
        #self.log.debug("new_gain_stage: {}".format(new_gain_stage))
        self.result["intervals"]["gain_stages"]["data"][self.gain_idx["low"]] = new_gain_stage


    def detect_saturation(self, interval):
        self.saturation_threshold = 14000

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

        self.result["intervals"]["saturation"]["data"][self.out_idx] = [sat_index, interval[1]]

        return [interval[0], sat_index]


    def detect_saturation2(self, interval):
        diff_det_interval = self.diff[interval[0]:interval[1]]

        sat_indices = np.where(np.absolute(diff_det_interval) < self.saturation_threshold)[0]

        j = sat_indices[-1]
        for i in sat_indices[::-1]:
            if i == j:
                j -= 1
            else:
                #self.log.debug("not true for {} {}".format(i, diff_det_interval[i-1:i+2]))
                break
        #self.log.debug("i {}".format(i))
        #self.log.debug("j {}".format(j))

        self.result["intervals"]["saturation"]["data"][self.out_idx] = [interval[0] + i, interval[1]]

        return [interval[0] + sat_indices[0], interval[0] + i]


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
        # meaning  data_to_fit = coefficient_matrix * [slope, offset]
        self.data_to_fit[gain][interval_idx] = self.analog[self.in_idx[0], self.in_idx[1],
                                                           self.in_idx[2],
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
        array_idx = self.out_idx + (interval_idx,)
        res = None
        try:
            res = np.linalg.lstsq(self.coefficient_matrix, self.data_to_fit[gain][interval_idx])

            self.result["slope"]["individual"][gain]["data"][array_idx] = res[0][0]
            self.result["offset"]["individual"][gain]["data"][array_idx] = res[0][1]
            self.result["residual"]["individual"][gain]["data"][array_idx] = res[1]
            self.result["average_residual"]["individual"][gain]["data"][array_idx] = (
                np.sqrt(res[1] / number_of_points))

            #self.log.debug("average_residual {}".format(self.result["average_residual"]["individual"][gain][array_idx]))
        except:
            if res is None:
                self.log.debug("interval\n{}".format(interval))
                self.log.debug("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
                self.log.debug("self.data_to_fit[{}][{}]\n{}"
                               .format(gain, interval_idx,
                                       self.data_to_fit[gain][interval_idx]))

                raise

            if res[0].size != 2:
                self.result["error_code"]["data"][self.out_idx] = 5
                raise FitError("Failed to calculate slope and offset")
            elif res[1].size != 1:
                self.result["warning_code"]["data"][self.out_idx] = 1
                raise FitError("Failed to calculate residual")
            else:
                self.log.debug("interval\n{}".format(interval))
                self.log.debug("self.coefficient_matrix\n{}".format(self.coefficient_matrix))
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
                   slice(self.result["intervals"]["gain_stages"]["data"][self.gain_idx[gain]][0],
                         self.result["intervals"]["gain_stages"]["data"][self.gain_idx[gain]][1]))

            self.result["medians"]["data"][self.gain_idx[gain]] = np.median(self.digital[idx])

    def calc_thresholds(self):
        # threshold between high and medium
        self.result["thresholds"]["data"][(0, ) + self.out_idx] = np.mean([
                self.result["medians"]["data"][self.gain_idx["high"]],
                self.result["medians"]["data"][self.gain_idx["medium"]]])

        # threshold between medium and low
        self.result["thresholds"]["data"][(1, ) + self.out_idx] = np.mean([
                self.result["medians"]["data"][self.gain_idx["medium"]],
                self.result["medians"]["data"][self.gain_idx["low"]]])

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

    # def write_data(self):
    #     save_file = h5py.File(self.out_fname, "w", libver="latest")

    #     try:
    #         self.log.info("\nStart saving data")
    #         t = time.time()

    #         for key in self.result:
    #             if type(self.result[key]) != dict:
    #                 save_file.create_dataset("/{}".format(key), data=self.result[key])
    #             else:
    #                 for subkey in self.result[key]:
    #                     if type(self.result[key][subkey]) != dict:
    #                         save_file.create_dataset("/{}/{}".format(key, subkey),
    #                                                  data=self.result[key][subkey])
    #                     else:
    #                         for gain in ["high", "medium", "low"]:
    #                             save_file.create_dataset("/{}/{}/{}".format(key, subkey, gain),
    #                                                      data=self.result[key][subke352n])

    #         save_file.flush()
    #         self.log.info("took time: {}".format(time.time() - t))
    #     finally:
    #         save_file.close()

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
