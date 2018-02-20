#!/usr/bin/python3

import os


class GeneratePathsXfel(object):
    def __init__(self,
                 run_type,
                 meas_type,
                 out_base_dir,
                 module,
                 channel,
                 temperature,
                 meas_spec,
                 meas_in,
                 asic,
                 runs,
                 run_name,
                 use_xfel_out_format):

        self.run_type = run_type
        self.meas_type = meas_type
        self.out_base_dir = out_base_dir
        self.module = module
        self.channel = channel
        self.temperature = temperature
        self.meas_spec = meas_spec
        self.meas_in = meas_in
        self.asic = asic
        self.runs = runs
        self.run_name = run_name

        self.use_xfel_out_format = use_xfel_out_format

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
            channel = str(self.channel).zfill(2)

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

        if self.meas_type == "pcdrs" or len(self.runs) == 1:
            run_subdir = "r" + "-r".join(str(r).zfill(4)
                                         for r in self.runs)

        else:
            print("WARNING: generate paths, preproc is running in 'else'. "
                  "Why?")
            print("self.runs={}".format(self.runs))
            run_subdir = "r{:04}".format(self.runs[0])

        if as_template:
            fdir = os.path.join(base_dir,
                                self.meas_type,
                                "r{run:04}")

            fname = "R{run:04}-preprocessing.result"
        else:
            fdir = os.path.join(base_dir,
                                self.meas_type,
                                run_subdir)

            fname = "R{:04}-preprocessing.result".format(self.runs[0])

        return fdir, fname

    def gather(self, base_dir):
        """Generate the gather file path.

        Args:
            base_dir: Base directory under which the output is stored.

        Return:
            Gather directory and file name, each as string.
        """

        # TODO: concider additing this into out_base_dir (joined) and
        #       create subdirs for gathered files
        if self.meas_type == "pcdrs" or len(self.runs) == 1:
            run_subdir = "r" + "-r".join(str(r).zfill(4)
                                         for r in self.runs)

            print("run_subdir", run_subdir)
            print("channel", self.channel)
            fname = ("{}-AGIPD{:02}-gathered.h5"
                     .format(run_subdir.upper(), self.channel))

        # TODO fill run_number + for what is this 'else' (what cases)?
        else:
            run_subdir = "r{run_number:04}"

            fname = ("R{run_number:04}-" +
                     "AGIPD{:02}-gathered.h5".format(self.channel))

        fdir = os.path.join(base_dir,
                            self.meas_type,
                            run_subdir,
                            "gather")

        return fdir, fname

    def process(self, base_dir, use_xfel_out_format, as_template=False):
        """Generate the process file path.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_out_format (bool): If enabled the output is in xfel format.
            as_template (optional, bool): If enabled the channel is kept as a
                                          template instead of being filled in
                                          (default: False).

        Return:
            Process directory and file name, each as string.
        """

        run_subdir = "r" + "-r".join(str(r).zfill(4) for r in self.runs)

        fdir = os.path.join(base_dir,
                            self.meas_type,
                            run_subdir,
                            "process")

        if use_xfel_out_format:
            fname = self.meas_type + "_AGIPD{:02}_xfel.h5"
        else:
            fname = self.meas_type + "_AGIPD{:02}_agipd.h5"

        if not as_template:
            fname = fname.format(self.channel)

        return fdir, fname

    def join(self, base_dir, use_xfel_out_format):
        """Generates the join file path.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_out_format (bool): If enabled the output is in xfel format.

        Return:
            Join directory and file name, each as string.
        """

        run_subdir = "r" + "-r".join(str(r).zfill(4) for r in self.runs)

        fdir = os.path.join(base_dir,
                            self.meas_type,
                            run_subdir)

        if use_xfel_out_format:
            fname = "{}_joined_constants_xfel.h5".format(self.meas_type)
        else:
            fname = "{}_joined_constants_agipd.h5".format(self.meas_type)

        return fdir, fname


class GeneratePathsCfel(object):
    def __init__(self,
                 run_type,
                 meas_type,
                 out_base_dir,
                 module,
                 channel,
                 temperature,
                 meas_spec,
                 meas_in,
                 asic,
                 runs,
                 run_name,
                 use_xfel_out_format):

        self.run_type = run_type
        self.meas_type = meas_type
        self.out_base_dir = out_base_dir
        self.module = module
        self.channel = channel
        self.temperature = temperature
        self.meas_spec = meas_spec
        self.meas_in = meas_in
        self.asic = asic
        self.runs = runs
        self.run_name = run_name

        self.use_xfel_out_format = use_xfel_out_format

    def raw(self, base_dir, as_template=True):
        """Generates raw path.

        Args:
            base_dir: Base directory under which the raw directory can be
                      found.
            as_template (optional, bool): If enabled the channel is kept as a
                                          template instead of being filled in.
                                          (default: True, but not implemented here)

        Return:
            Raw directory and file name, each as string. The file name as
            a wildcard for the channel.
        """

        # as_template refers to module/channel

        # define in files
        fdir = os.path.join(base_dir,
                            self.temperature,
                            self.meas_in[self.meas_type])

        if self.meas_type not in ["dark", "xray"]:
            fdir = os.path.join(base_dir,
                                self.meas_spec)

        #fname = ("{}_{}_{}_"
        fname = ("{}*_{}_{}_"  # only module without location, e.g. M304
                 .format(self.module,
                         self.meas_type,
                         self.meas_spec)
                 + "{run_number:05}_part{part:05}.nxs")

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

    def gather(self, base_dir):
        """Generate the gather file path.

        Args:
            base_dir: Base directory under which the output is stored.

        Return:
            Gather directory and file name, each as string.
        """

        # define out files
        fdir = os.path.join(base_dir,
                            self.module,
                            self.temperature,
                            self.meas_type,
                            self.meas_spec,
                            self.run_type)
        # for testing
#        out_base_dir = ("/gpfs/exfel/exp/SPB/201701/p002012/" +
#                        "scratch/user/kuhnm")
#        out_subdir = "tmp"
#        out_dir = os.path.join(out_base_dir,
#                               out_subdir,
#                               "gather")
        if self.asic is None:
            fname = ("{}_{}_{}_gathered.h5"
                     .format(self.module,
                             self.meas_type,
                             self.run_name))
        else:
            fname = ("{}_{}_{}_asic{:02}_gathered.h5"
                     .format(self.module,
                             self.meas_type,
                             self.run_name,
                             self.asic))

        return fdir, fname

    def process(self, base_dir, use_xfel_out_format, as_template=False):
        """Generate the process file path.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_out_format (bool): If enabled the output is in xfel format
                                        (not implemented).
            as_template (optional, bool): If enabled the channel is kept as a
                                          template instead of being filled in
                                          (default: False, but not implemented here)

        Return:
            Process directory and file name, each as string.
        """

        fdir = os.path.join(self.out_base_dir,
                            self.module,
                            self.temperature,
                            self.meas_type,
                            self.meas_spec,
                            self.run_type)

        if self.asic is None:
            fname = ("{}_{}_{}_processed.h5"
                     .format(self.module,
                             self.meas_type,
                             self.meas_spec))
        else:
            fname = ("{}_{}_{}_asic{:02}_processed.h5"
                     .format(self.module,
                             self.meas_type,
                             self.meas_spec,
                             self.asic))

        print("process fname", fname)

        return fdir, fname

    def join(self, base_dir, use_xfel_out_format):
        """Generates the join file path. NOT IMPLEMENTED.

        Args:
            base_dir: Base directory under which the output is stored.
            use_xfel_out_format (bool): If enabled the output is in xfel format.

        Return:
            Join directory and file name, each as string.
        """
        raise Exception("CFEL gpfs not supported for join at the moment")
#        fname = ("{}_{}_{}_asic{:02d}_processed.h5"
#                 .format(self.module,
#                         self.meas_type,
#                         self.meas_spec,
#                         self.asic))

#        print("process fname", fname)

#        fdir = os.path.join(self.out_base_dir,
#                            self.module,
#                            self.temperature,
#                            self.meas_type,
#                            self.meas_spec,
#                            self.run_type)

#        return fdir, fname


