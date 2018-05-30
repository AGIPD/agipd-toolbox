import os
import sys
import time

BASE_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
SRC_PATH = os.path.join(BASE_PATH, "src")

if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import utils  # noqa E402


class JobOverview(object):
    def __init__(self, rtl, panel_list, enable_asics=False):
        self.enable_asics = enable_asics
        print("enable_asics", enable_asics)

        self.dep_overview = {}

        # jobs concering all panels (before single panel jobs are done)
        self.dep_overview["all_before"] = {}
        for run_type in rtl.panel_dep_before:
            self.dep_overview["all_before"][run_type] = {}

        # jobs concering single panel
        for panel in panel_list:
            panel = str(panel)

            self.dep_overview[panel] = {}

            for run_type in rtl.per_panel:
                self.dep_overview[panel][run_type] = {}

        # jobs concering all panels (after single panel jobs were done)
        self.dep_overview["all_after"] = {}
        for run_type in rtl.panel_dep_after:
            self.dep_overview["all_after"][run_type] = {}

    def fill(self, group_name, runs, asics, run_type, jn, jobs):
        """Filling content into the overview dictionary.

        Args:
            group_name: The group name this information belongs to,
                        e.g. 'all_before', a specific panel or 'all_after'.
            runs (list): A list of runs
            asics (list or None): A list of asics or None if asic splitting
                                  was disabled.
            run_type (str): The run type of the job,
                            e.g. 'gather', 'process', ...
            jn: Under which job number the job was launched.
            jobs: The jobs this job is depending on.
        """

        if asics is None:
            asics_str = "all"
        else:
            asics_str = "-".join(map(str, asics))

        group_name = str(group_name)

        if type(runs) == list:
            runs_string = "-".join(list(map(str, runs)))
        else:
            runs_string = str(runs)

        ov_r = self.dep_overview[group_name][run_type]
        if runs_string in ov_r:
            ov_r[runs_string][asics_str] = {}
        else:
            ov_r[runs_string] = {asics_str: {}}

        ov_r[runs_string][asics_str]["jobnum"] = jn
        ov_r[runs_string][asics_str]["deb_jobs"] = jobs

    def prepare(self, header, sep):
        """Preparation for printing.

        Determines the column sizes and sorts the keys.

        Args.
            header (list): The columns names of the overview table.
            sep (str): The seperater to be used to distinguish the columns.
        """

        # determine how long the space for the keys should be
        max_key_len = {}
        for key in header:
            max_key_len[key] = len(key)

        for panel in self.dep_overview:
            max_key_len["Panel"] = max(max_key_len["Panel"], len(panel))

            for run_type in self.dep_overview[panel]:
                max_key_len["Run type"] = max(max_key_len["Run type"],
                                              len(run_type))

                for runs in self.dep_overview[panel][run_type]:
                    max_key_len["Runs"] = max(max_key_len["Runs"], len(runs))

                    for asics in self.dep_overview[panel][run_type][runs]:
                        try:
                            max_key_len["Asics"] = max(max_key_len["Asics"],
                                                       len(asics))
                        except KeyError:
                            pass

        # job ids are 6-digit numbers
        max_key_len["JobID"] = max(max_key_len["JobID"], 6)

        # fill up header with spaces according to max length of strings in
        # column
        new_header = []
        for i, key in enumerate(header):
            new_key = key.ljust(max_key_len[key])
            new_header.append(new_key)

        # sort the rows
        key_list = [key for key in self.dep_overview
                    if not key.startswith("all")]
        sorted_keys = sorted(key_list)
        # to sort the list numerically the entries have to be ints
        # but to access the dictionary they have to be converted back to string
        sorted_keys = list(map(str, sorted_keys))
        sorted_keys = ["all_before"] + sorted_keys + ["all_after"]

        # generate header
        header_str = sep.join(list(map(str, new_header)))
        header_str += "\n"

        # add separation between column description and rows
        for key in header:
            value = max_key_len[key]
            header_str += "-" * value + sep

        return max_key_len, header_str, sorted_keys

    def print(self):
        """Print overview of dependencies.
        """

        if self.enable_asics:
            header = ["Panel", "Run type", "Runs", "Asics", "JobID", "Dependencies"]
        else:
            header = ["Panel", "Run type", "Runs", "JobID", "Dependencies"]
        sep = " "

        max_key_len, header_str, sorted_keys = self.prepare(header, sep)

        print("\nDependencies Overview")
        print(header_str)

        # print overview
        for panel in sorted_keys:
            for run_type in self.dep_overview[panel]:
                for runs in self.dep_overview[panel][run_type]:
                    for asics in self.dep_overview[panel][run_type][runs]:
                        d_o = self.dep_overview[panel][run_type][runs][asics]

                        col = []
                        col.append(panel.ljust(max_key_len[header[0]]))
                        col.append(run_type.ljust(max_key_len[header[1]]))
                        col.append(runs.ljust(max_key_len[header[2]]))

                        if self.enable_asics:
                            col.append(asics.ljust(max_key_len[header[3]]))
                            col.append(str(d_o["jobnum"]).ljust(max_key_len[header[4]]))
                        else:
                            col.append(str(d_o["jobnum"]).ljust(max_key_len[header[3]]))

                        if d_o["deb_jobs"] == "":
                            col.append("no dependencies")
                        else:
                            col.append(str(d_o["deb_jobs"]))

                        print(sep.join(col))

    def get():
        """Get the content of the overview dictionary.

        Return
            The overview dictionary.
        """

        return self.overview

    def generate_mail_body(self):

        header = ["Panel", "Run type", "Runs", "JobID", "State"]
        sep = " "

        overview = self.get()

        time.sleep(3)
        # get status of jobs
        status = {}
        for panel in overview:
            for run_type in overview[panel]:
                for runs in overview[panel][run_type]:
                    d_o = overview[panel][run_type][runs]

                    if d_o["jobnum"] is not None:
                        cmd = ["sacct", "--brief", "-p", "-j", d_o["jobnum"]]
#                        os.system("squeue --user $USER")
                        result = utils.submit_job(cmd, jobname="sacct")
                        result = result.split()

                        # the line ends with a separator
                        # -> remove last character
#                        status_header = result[0][:-1].split("|")
#                        print(status_header)
                        for res in result[1:]:
                            # the line ends with a separator
                            status_result = res[:-1].split("|")
#                            print("status_result", status_result)
                            # sacct give the two results for every job
                            # <jobid> and <jobid>.batch
                            if status_result[0] == d_o["jobnum"]:
                                # status results as the entries
                                # 'JobID', 'State', 'ExitCode'
                                status[d_o["jobnum"]] = status_result[1]

        max_key_len, header_str, sorted_keys = self.overview.prepare(header, sep)

        print("\nMail Body")
        print(header_str)

        # print overview
        for panel in sorted_keys:
            for run_type in overview[panel]:
                for runs in overview[panel][run_type]:
                    d_o = overview[panel][run_type][runs]

                    row = []
                    row.append(panel.ljust(max_key_len[header[0]]))
                    row.append(run_type.ljust(max_key_len[header[1]]))
                    row.append(runs.ljust(max_key_len[header[2]]))
                    row.append(str(d_o["jobnum"]).ljust(max_key_len[header[3]]))
                    row.append(status[d_o["jobnum"]])

                    print(sep.join(row))


