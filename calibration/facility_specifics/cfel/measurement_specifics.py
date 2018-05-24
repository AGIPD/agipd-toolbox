import sys
from __init__ import FACILITY_DIR

if FACILITY_DIR not in sys.path:
    sys.path.insert(0, FACILITY_DIR)


class Measurement(object):
    def __init__(self):
        self.meas_spec = None

    def get_safety_factor(self, config):
        """Get the safety factor which is used in process

        Args:
            config (dict): The dictionary created when reading the config file.

        Return:
            An int read in the measurement specific entry in the config.
        """
        return None

    def get_meas_spec(self, config):
        """Get the measurement depending specific tag.

        Args:
            config (dict): The dictionary created when reading the config file.

        Return:
            A string read in the measurement specific entry in the config.
        """

        return [None]

    def get_subdir(self, config):
        """Get the subdir if one is configured.

        Args:
            config (dict): The dictionary created when reading the config file.

        Return:
            A string read in the subdir entry in the config.
            Is set to None if nothing was configured.
        """

        try:
            subdir = config[self.measurement]["subdir"]
        except KeyError:
            return None

        if subdir == False:
            subdir = None

        return subdir

    def get_run_type_list(self):
        """ Get the list of steps to be taken to get the constants.

        Return:
            A list of run_types, e.g. ["gather", "process"].
        """

        run_type_list = ["gather", "process"]

        return run_type_list

    def get_run_number_templ(self):
        """ Get the template for the run number used in the file name.

        This method only works in CFEL mode.

        Return:
            A string containing the template and a int describing how many
            many splits to be taken to determine the run number.
        """
        run_number_templ = "{run_number:05}"
        split_number = 1

        return run_number_templ, split_number

    def mod_params(self, param_list, run_type):
        """ Add measurement specific parameters.

        Args:
            param_list (list): A list with the already set parameters.
            run_type (str): Which run type is used.

        Return:
            A list with the modified parameters.
        """
        if self.meas_spec is not None:
            param_list += ["--meas_spec", self.meas_spec]

        return param_list


class Dark(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "dark"

    def get_meas_spec(self, config):
        """Get the measurement depending specific tag.

        Args:
            config (dict): The dictionary created when reading the config file.

        Return:
            A string read in the measurement specific entry in the config.
        """
        tint = config[self.measurement]["tint"]
        self.meas_spec = [tint]

        return self.meas_spec


class Drspc(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "drspc"


class Drscs(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "drscs"

    def get_safety_factor(self, config):
        return config[self.measurement]["safety_factor"]

    def get_meas_spec(self, config):
        """Get the measurement depending specific tag.

        Args:
            config (dict): The dictionary created when reading the config file.

        Return:
            A string read in the measurement specific entry in the config.
        """

        c_current = config[self.measurement]["current"]

        # comma seperated string into into list
        current_list = [c.split()[0] for c in c_current.split(",")]
        self.measurement = current_list

        return self.meas_spec

    def get_run_type_list(self):
        """ Get the list of steps to be taken to get the constants.

        Return:
            A list of run_types, e.g. ["gather", "process"]
        """

        run_type_list = ["gather", "process", "merge"]

        return run_type_list

    def get_run_number_templ(self):
        """ Get the template for the run number used in the file name.

        This method only works in CFEL mode.

        Return:
            A string containing the template and a int describing how many
            many splits to be taken to determine the run number.
        """
        run_number_templ = self.meas_spec + "_{run_number:05}"
        split_number = 2

        return run_number_templ, split_number

    def mod_params(self, param_list, run_type):
        """ Add measurement specific parameters.

        Args:
            param_list (list): A list with the already set parameters.
            run_type (str): Which run type is used.

        Return:
            A list with the modified parameters.
        """

        currents = "-".join(self.meas_spec)
        self.script_params += ["--current_list", currents]

        return param_list


class Xray(Measurement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.measurement = "xray"

    def get_meas_spec(self, config):
        """Get the measurement depending specific tag.

        Args:
            config (dict): The dictionary created when reading the config file.

        Return:
            A string read in the measurement specific entry in the config.
        """

        element = config[self.measurement]["element"]
        self.meas_spec = [element]

        return self.meas_spec
