#!/usr/bin/python3

import h5py
import os


class GeneratePaths(object):
    def __init__(self,
                 run_type,
                 measurement,
                 out_base_dir,
                 module,
                 channel,
                 temperature,
                 meas_spec,
                 meas_in,
                 asic,
                 runs,
                 run_name):

        self._run_type = run_type
        self._measurement = measurement
        self._out_base_dir = out_base_dir
        self._module = module
        self._channel = channel
        self._temperature = temperature
        self._meas_spec = meas_spec
        self._meas_in = meas_in
        self._asic = asic
        self._runs = runs
        self._run_name = run_name

    def get_layout_versions(self, base_dir):
        """ Detects which file structure version of the raw files.

        Args:
            base_dir: Base directory under which the raw directory can be
                      found.

        Return:
            The preprocess and layout module to use.
        """
        return None, "cfel_layout"

    def raw(self, base_dir, as_template=True):
        """Generates raw path.

        Args:
            base_dir: Base directory under which the raw directory can be
                      found.
            as_template (optional, bool): If enabled the channel is kept as a
                                          template instead of being filled in.
                                          (default: True,
                                                    but not implemented here)

        Return:
            Raw directory and file name, each as string. The file name as
            a wildcard for the channel.
        """

        # as_template refers to module/channel

        # define in files
        fdir = os.path.join(base_dir,
                            self._temperature,
                            self._meas_in[self._measurement])

        prefix = ("{}*_{}_"  # only module without location, e.g. M304
                  .format(self._module,
                          self._measurement))

        if self._meas_spec is not None:
            if self._measurement not in ["dark", "xray"]:
                fdir = os.path.join(fdir, self._meas_spec)

            prefix += "{}_".format(self._meas_spec)

        if self._run_name is None:
            fname = prefix + "{run_number:05}_part{part:05}.nxs"
        elif type(self._run_name) == str:
            fname = (prefix
                     + self._run_name
                     + "_{run_number:05}_part{part:05}.nxs")
#        elif len(self._run_name) == 1:
#            fname = (prefix
#                     + self._run_name[0]
#                     + "_{run_number:05}_part{part:05}.nxs")
        elif (type(self._run_name) == list
                and self._run_name
                and type(self._run_name[0]) != list):
            fname = (prefix
                     + "{run_name}"
                     + "_{run_number:05}_part{part:05}.nxs")

        else:
            print("run_name", self._run_name)
            raise Exception("Run name is not unique.")

        return fdir, fname

    def preproc(self, base_dir, as_template=False):
        """Generates the preprocessing file path. NOT IMPLEMENTED.

        Args:
            base_dir: Base directory under which the output is stored.
            as_template (optional, bool): If enabled the run is kept as a
                                          template instead of being filled in
                                          (default: False).

        Return:
            None, None because preprocessing is not implemented in the CFEL
            layout.
        """

        print("Preprocessing not implemented for CFEL layout")

        return None, None

    def gather(self, base_dir, as_template=False):
        """Generate the gather file path.

        Args:
            base_dir: Base directory under which the output is stored.

        Return:
            Gather directory and file name, each as string.
        """

        # define out files
        fdir = os.path.join(base_dir,
                            self._module,
                            self._temperature,
                            self._measurement)
        if self._meas_spec:
            fdir = os.path.join(fdir, self._meas_spec)

        fdir = os.path.join(fdir, "gather")

        if self._run_name is None:
            prefix = ("{}_{}"
                      .format(self._module,
                              self._measurement))

        else:
            if as_template:
                name = "{run_name}"
            else:
                name = "-".join(self._run_name)

            prefix = ("{}_{}_{}"
                      .format(self._module,
                              self._measurement,
                              name))

        if self._asic is None:
            fname = prefix + "_gathered.h5"
        else:
            fname = prefix + "_asic{:02}_gathered.h5".format(self._asic)

        return fdir, fname

    def process(self, base_dir, use_xfel_format, as_template=False):
        """Generate the process file path.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_format (bool): If enabled the output is in xfel format
                                    (not implemented).
            as_template (optional, bool): If enabled the channel is kept as a
                                          template instead of being filled in
                                          (default: False,
                                                    but not implemented here)

        Return:
            Process directory and file name, each as string.
        """

        # set the directory

        fdir = os.path.join(self._out_base_dir,
                            self._module,
                            self._temperature,
                            self._measurement)

        if self._meas_spec is not None:
            fdir = os.path.join(fdir, self._meas_spec)

        fdir = os.path.join(fdir, self._run_type)

        # set the file name

        if self._asic is None:
            postfix = "processed.h5"
        else:
            postfix = "asic{:02}_processed.h5".format(self._asic)

        if self._meas_spec is not None:
            postfix = "{}_{}".format(self._meas_spec, postfix)

        fname = ("{}_{}_{}"
                 .format(self._module,
                         self._measurement,
                         postfix))

        print("process fname", fname)

        return fdir, fname

    def join(self, base_dir, use_xfel_format):
        """Generates the join file path. NOT IMPLEMENTED.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_format (bool): If enabled the output is in xfel
                                    format.

        Return:
            Join directory and file name, each as string.
        """
        raise Exception("CFEL gpfs not supported for join at the moment")
