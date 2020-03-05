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
                 subdir,
                 meas_in,
                 asic,
                 runs,
                 run_name,
                 detector_string):

        self._run_type = run_type
        self._measurement = measurement
        self._out_base_dir = out_base_dir
        self._module = module
        self._channel = channel
        self._temperature = temperature
        self._meas_spec = meas_spec
        self._subdir = subdir
        self._meas_in = meas_in
        self._asic = asic
        self._runs = runs
        self._run_name = run_name
        self._detector_string = detector_string

    def get_layout_versions(self, base_dir):
        """ Detects which file structure version of the raw files.

        Args:
            base_dir: Base directory under which the raw directory can be
                      found.

        Return:
            The preprocess and layout module to use.
        """

        # current version can only be detected by checking if certain entries
        # are contained in the hdf5 file because there is no versioning

        if type(self._channel) == list:
            fdir, fname = self.raw(base_dir=base_dir, as_template=True)
            raw_fname = os.path.join(fdir, fname)

            # partially substitute the string
            split = raw_fname.rsplit("-AGIPD{:02}", 1)
            raw_fname = (
                split[0] +
                "-AGIPD{:02}".format(self._channel[0]) +
                split[1]
            )

            channel = self._channel[0]
        else:
            fdir, fname = self.raw(base_dir=base_dir)
            raw_fname = os.path.join(fdir, fname)

            channel = self._channel

        entry_to_test = (
            "INDEX/{}/DET/{}CH0:xtdf/image/last"
            .format(self._detector_string, channel)
        )

        raw_fname = raw_fname.format(run_number=self._runs[0], part=0)

        with h5py.File(raw_fname, "r") as f:
            try:
                f[entry_to_test]
                print("found")
                version = 2017
            except KeyError:
                print("Entry {} not found, version = 2018".format(entry_to_test))
                version = 2018

        if version == 2017:
            return "xfel_preprocess_2017", "xfel_layout_2017"
        else:
            return "xfel_preprocess", "xfel_layout"

    def raw(self, base_dir, as_template=False):
        """Generates raw path.

        Args:
            base_dir: Base directory under which the raw directory can be
                      found.
            as_template (optional, bool): If enabled the channel is kept as a
                                          template instead of being filled in
                                          (default: False).

        Return:
            Raw directory and file name, each as string.
        """

        if as_template:
            channel = "{:02}"
        else:
            channel = str(self._channel).zfill(2)

        # define in files
        fdir = os.path.join(base_dir,
                            "raw",
                            "r{run_number:04}")

        fname = ("RAW-R{run_number:04}-" +
                 "AGIPD{}".format(channel) +
                 "-S{part:05}.h5")

        return fdir, fname

    def preproc(self, base_dir, as_template=False):
        """Generates the preprocessing file path.

        Args:
            base_dir: Base directory under which the output is stored.
            as_template (optional, bool): If enabled the run is kept as a
                                          template instead of being filled in
                                          (default: False).

        Return:
            Preprocessing directory and file name, each as
            string.
        """

        if self._measurement == "drspc" or len(self._runs) == 1:
            run_subdir = "r" + "-r".join(str(r).zfill(4)
                                         for r in self._runs)

        else:
            print("WARNING: generate paths, preproc is running in 'else'. "
                  "Why?")
            print("self._runs={}".format(self._runs))
            run_subdir = "r{:04}".format(self._runs[0])

        if as_template:
            fdir = os.path.join(base_dir,
                                self._measurement,
                                "r{run:04}")

            fname = "R{run:04}-preprocessing.result"
        else:
            fdir = os.path.join(base_dir,
                                self._measurement,
                                run_subdir)

            fname = "R{:04}-preprocessing.result".format(self._runs[0])

        return fdir, fname

    def gather(self, base_dir, as_template=False):
        """Generate the gather file path.

        Args:
            base_dir: Base directory under which the output is stored.

        Return:
            Gather directory and file name, each as string.
        """

        # TODO: concider additing this into out_base_dir (joined) and
        #       create subdirs for gathered files
        if self._measurement == "drspc" or len(self._runs) == 1:
            run_subdir = "r" + "-r".join(str(r).zfill(4)
                                         for r in self._runs)

            print("run_subdir", run_subdir)
            print("channel", self._channel)
            prefix = ("{}-AGIPD{:02}"
                      .format(run_subdir.upper(), self._channel))

        # TODO fill run_number + for what is this 'else' (what cases)?
        else:
            run_subdir = "r{run_number:04}"

            prefix = ("R{run_number:04}-" +
                      "AGIPD{:02}".format(self._channel))

        fdir = os.path.join(base_dir,
                            self._measurement,
                            run_subdir,
                            "gather")


        if self._asic is None:
            fname = prefix + "_gathered.h5"
        else:
            fname = prefix + "_asic{:02}_gathered.h5".format(self._asic)

        return fdir, fname

    def process(self, base_dir, use_xfel_format, as_template=False):
        """Generate the process file path.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_format (bool): If enabled the output is in xfel
                                    format.
            as_template (optional, bool): If enabled the channel is kept as a
                                          template instead of being filled in
                                          (default: False).

        Return:
            Process directory and file name, each as string.
        """

        run_subdir = "r" + "-r".join(str(r).zfill(4) for r in self._runs)

        fdir = os.path.join(base_dir,
                            self._measurement,
                            run_subdir,
                            "process")

        prefix = self._measurement + "-AGIPD{:02}"

        if self._asic is not None:
            prefix = prefix + "_asic{:02}"

        if use_xfel_format:
            fname = prefix + "_xfel.h5"
        else:
            fname = prefix + "_agipd.h5"

        if not as_template:
            fname = fname.format(self._channel, self._asic)

        return fdir, fname

    def join(self, base_dir, use_xfel_format):
        """Generates the join file path.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_format (bool): If enabled the output is in xfel
                                    format.

        Return:
            Join directory and file name, each as string.
        """

        run_subdir = "r" + "-r".join(str(r).zfill(4) for r in self._runs)

        fdir = os.path.join(base_dir,
                            self._measurement,
                            run_subdir)

        if use_xfel_format:
            fname = "{}_joined_constants_xfel.h5".format(self._measurement)
        else:
            fname = "{}_joined_constants_agipd.h5".format(self._measurement)

        return fdir, fname

