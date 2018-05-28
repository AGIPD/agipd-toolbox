from collections import namedtuple
import glob
import json
import os
import sys

from generate_paths import GeneratePaths


class RunType(object):
    Rtl = namedtuple("rtl", ["panel_dep_before",
                             "per_panel",
                             "panel_dep_after"])

    def __init__(self):
        self.run_type = None

    def get_channel_list(self, l):
        return []

    def get_module_list(self, l):
        if type(l) == list:
            return l
        else:
            return [l]

    def get_temperature(self, config):
        return config

    def get_max_part(self, config):
        return None

    def get_run_list(self,
                     c_run_list,
                     measurement,
                     module_list,
                     channel_list,
                     temperature,
                     meas_spec,
                     subdir,
                     input_dir,
                     meas_conf,
                     run_name):
        return []

    def _get_cfel_run_list(self,
                           measurement,
                           module_list,
                           temperature,
                           meas_spec,
                           subdir,
                           input_dir,
                           meas_conf,
                           run_name):

        generate_paths = GeneratePaths(
            run_type=None,
            measurement=measurement,
            out_base_dir=None,
            module=module_list[0],
            channel=None,
            temperature=temperature,
            meas_spec=meas_spec[0],
            subdir=subdir,
            meas_in={measurement: measurement},
            asic=None,
            runs=None,
            run_name=None
        )

        raw_dir, raw_fname = generate_paths.raw(input_dir['gather'])
        raw_path = os.path.join(raw_dir, raw_fname)
#        print("raw_path", raw_path)

        run_number_templ, split_number = meas_conf.get_run_number_templ()
#        print("run_number_templ", run_number_templ)

        # we are trying to determine the run_number, thus we cannot fill it in
        # and have to replace it with a wildcard
        raw = raw_path.replace(run_number_templ, "*")

        raw = raw.format(part=0)
#        print("raw", raw)

        found_files = glob.glob(raw)
#        print("found_files", found_files)

        raw_splitted = raw_path.split(run_number_templ)

        postfix = raw_splitted[-1]
        postfix = postfix.format(part=0)

        if not found_files:
            print("Searched for file matching:", raw)
            raise Exception("No raw files found.")

        run_numbers_dict = {}
        for f in found_files:

            # also remove underscore
            middle_part = f[len(raw_splitted[0]) + 2:]

            # only take the runs for which run_names are defined
            if run_name is None or middle_part.startswith(tuple(run_name)):
                # cut off the part after the run_number
                rnumber = f[:-len(postfix)]
                print("rnumber", rnumber)
                # the run number now is the tail of the string till the
                # underscore (for drscs it is not the last but the second
                # to last underscore)
                rnumber = rnumber.rsplit("_", split_number)[1:]
                print("rnumber", rnumber)

                # for drscs the splitted elements have to be join again
                rnumber = "_".join(rnumber)
                print("rnumber", rnumber)

                # which run name the run number matches
                rname = [name for name in run_name
                         if middle_part.startswith(name)][0]
                print("rname", rname)

                if measurement == "drscs":
                    run_numbers_dict[rname] = rnumber
                else:
                    run_numbers_dict[rname] = int(rnumber)

        print("run_numbers_dir", run_numbers_dict)

        if not run_numbers_dict:
            raise Exception("No file matching run_names found.")

        # order the run_numbers the same way as the run names
        # the order in the run names is important to determine the different
        # gain stages
        run_numbers = []
        for name in run_name:
            if name not in run_numbers_dict:
                raise Exception("run_name {} not found.".format(name))
            run_numbers.append(run_numbers_dict[name])
        print("run_numbers", run_numbers)

        if not run_numbers:
            print("raw:", raw)
            raise Exception("ERROR: No runs found.")

        return run_numbers

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = []
        rtl_per_panel = [self.run_type]
        rtl_panel_dep_after = []

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)

    def get_list_and_name(self, measurement, run_list, run_name, run_type):
        new_run_list = [run_list]
        new_run_name = [run_name]

        return new_run_list, new_run_name


class Preprocess(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "preprocess"

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = [self.run_type]
        rtl_per_panel = []
        rtl_panel_dep_after = []

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)


class Gather(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "gather"

    def get_max_part(self, config):
        try:
            return config["gather"]["max_part"]
        except:
            return super().get_max_part(config)

    def get_run_list(self,
                     c_run_list,
                     measurement,
                     module_list,
                     channel_list,
                     temperature,
                     meas_spec,
                     subdir,
                     input_dir,
                     meas_conf,
                     run_name):

        if c_run_list is None:
            return self._get_cfel_run_list(measurement=measurement,
                                           module_list=module_list,
                                           temperature=temperature,
                                           meas_spec=meas_spec,
                                           subdir=subdir,
                                           input_dir=input_dir,
                                           meas_conf=meas_conf,
                                           run_name=run_name)
        else:
            return c_run_list

    def get_list_and_name(self, measurement, run_list, run_name, run_type):
        if measurement == "dark":
            return run_list, run_name
        else:
            return super().get_list_and_name(measurement=measurement,
                                             run_list=run_list,
                                             run_name=run_name,
                                             run_type=run_type)


class Process(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "process"


class Merge(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "merge"


class Join(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_type = "join"

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = []
        rtl_per_panel = []
        rtl_panel_dep_after = [self.run_type]

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)


class All(RunType):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_run_list(self,
                     c_run_list,
                     measurement,
                     module_list,
                     channel_list,
                     temperature,
                     meas_spec,
                     subdir,
                     input_dir,
                     meas_conf,
                     run_name):

        if c_run_list is None:
            return self._get_cfel_run_list(measurement=measurement,
                                           module_list=module_list,
                                           temperature=temperature,
                                           meas_spec=meas_spec,
                                           subdir=subdir,
                                           input_dir=input_dir,
                                           meas_conf=meas_conf,
                                           run_name=run_name)
        else:
            return c_run_list

    def get_list_and_name(self, measurement, run_list, run_name, run_type):
        if run_type == "gather" and measurement == "dark":
            return run_list, run_name
        else:
            return super().get_list_and_name(measurement=measurement,
                                             run_list=run_list,
                                             run_name=run_name,
                                             run_type=run_type)

    def get_run_type_lists_split(self, run_type_list):
        rtl_panel_dep_before = []
        rtl_per_panel = run_type_list
        rtl_panel_dep_after = []

        return RunType.Rtl(panel_dep_before=rtl_panel_dep_before,
                           per_panel=rtl_per_panel,
                           panel_dep_after=rtl_panel_dep_after)
