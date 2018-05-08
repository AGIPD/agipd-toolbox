import os
import numpy as np
from plotting import GeneratePlots, generate_failed_images_plot
import argparse
from string import Template
import h5py


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_dir",
                        type=str,
                        default="/gpfs/cfel/fsds/labs/agipd/calibration/"
                                "processed/",
                        help="Processing directory base")
    parser.add_argument("--n_processes",
                        type=int,
                        default=10,
                        help="The number of processes for the pool")
    parser.add_argument("--module",
                        type=str,
                        required=True,
                        help="Module to gather, e.g M310")
    parser.add_argument("--temperature",
                        type=str,
                        required=True,
                        help="temperature to gather, e.g. temperature_30C")
    parser.add_argument("--current",
                        type=str,
                        help="Current to use, e.g. itestc20")
    parser.add_argument("--plot_dir",
                        type=str,
                        help="Subdir in which the plots should be stored")

    parser.add_argument("--merged",
                        action="store_true",
                        help="Use the merged processing results")
    parser.add_argument("--no_plotting",
                        action="store_true",
                        help="Disable plot generation")
    parser.add_argument("--no_over_plotting",
                        action="store_true",
                        help="Does not create already created plots")

    args = parser.parse_args()

    return args


def create_dir(directory_name):
    if not os.path.exists(directory_name):
        try:
            os.makedirs(directory_name)
            print("Dir '{0}' does not exist. Create it."
                  .format(directory_name))
        except IOError:
            if os.path.isdir(directory_name):
                pass


class GenerateStats(object):
    def __init__(self, module, current, gather_template, process_template,
                 plot_prefix, plot_dir_template, n_processes, stat_fname,
                 no_plotting, no_over_plotting):

        self.module = module
        self.current = current
        self.gather_template = gather_template
        self.process_template = process_template
        self.plot_prefix = plot_prefix
        self.plot_dir_template = plot_dir_template
        self.n_processes = n_processes
        self.stat_fname = stat_fname
        self.no_plotting = no_plotting
        self.no_over_plotting = no_over_plotting
        self.stats = dict()

        self.plot_ending = ".png"

        self.asic_list = range(1, 17)

        self.n_mem_cells = 352
        self.n_pixels_v = 64
        self.n_pixels_u = 64

    def run(self):

        self.write_preamble()

        for asic in self.asic_list:
            process_fname = process_template.substitute(a=str(asic).zfill(2))
            indices, occurences, total_number = (
                self.determine_failed_pixels(process_fname, asic))

            self.calc_stats(asic)

            plot_dir = self.plot_dir_template.substitute(a=str(asic).zfill(2))

            if not self.no_plotting:
                create_dir(plot_dir)

                gather_template = (self.gather_template
                                   .safe_substitute(a=str(asic).zfill(2)))
                gather_template = Template(gather_template)

                obj = GeneratePlots(asic,
                                    self.current,
                                    gather_template,
                                    self.plot_prefix,
                                    plot_dir,
                                    self.n_processes)

                current_value = int(self.current[len("itestc"):])
                for i in np.arange(len(indices[0])):
                    idx = (indices[0][i], indices[1][i], indices[2][i])

                    if self.no_over_plotting:
                        file_name = ("{}_asic{}_[{}, {}]_{}_data{}"
                                     .format(self.plot_prefix,
                                             str(asic).zfill(2),
                                             idx[0], idx[1],
                                             str(idx[2]).zfill(3),
                                             self.plot_ending))
                        plot_fname = os.path.join(plot_dir, file_name)

                        if not os.path.exists(plot_fname):
                            obj.run_idx(idx, current_value)
                    else:
                        obj.run_idx(idx, current_value)
                    #    obj.run_condition(process_fname, condition_function)

            plot_dir_images = os.path.join(plot_dir,
                                           "failed_pixels_per_images")
            create_dir(plot_dir_images)

            plot_file_name = self.plot_image_loss(asic, plot_dir_images)

            self.write_asic_table(asic,
                                  indices,
                                  occurences,
                                  total_number,
                                  plot_dir)

            self.write_asic_summary(asic, plot_dir_images, plot_file_name)

        self.write_epilog()

    def determine_failed_pixels(self, process_fname, asic):

        with h5py.File(process_fname, "r") as f:
            error_code = f["/error_code"][()]

        self.stats[asic] = dict()

        s = self.stats[asic]

        indices = np.where(error_code != 0)
        total_number = indices[0].size

        s["indices"] = indices
        s["total_number"] = total_number

        prev = (0, 0)
        result_indices = ([], [], [])
        occurrences = []
        for i in np.arange(indices[0].size):
            if indices[0][i] == prev[0] and indices[1][i] == prev[1]:
                occurrences[-1] += 1
            else:
                result_indices[0].append(indices[0][i])
                result_indices[1].append(indices[1][i])
                result_indices[2].append(indices[2][i])
                occurrences.append(1)

                prev = (indices[0][i], indices[1][i])

        return result_indices, occurrences, total_number

    def calc_stats(self, asic):
        indices = self.stats[asic]["indices"]

        failed_images = np.zeros(self.n_mem_cells, dtype=np.int)
        for i in np.arange(indices[2].size):
            failed_images[indices[2][i]] += 1

        self.stats[asic]["failed_images"] = failed_images

    def plot_image_loss(self, asic, plot_dir):

        plot_file_prefix = "{}_asic{}".format(self.plot_prefix,
                                              str(asic).zfill(2))
        plot_file_prefix = os.path.join(plot_dir, plot_file_prefix)

        plot_title_prefix = plot_file_prefix.rsplit("/", 1)[1]

        plot_title = "{} failed pixels per image".format(plot_title_prefix)
        plot_name = "{}_failed_pixels_per_image{}".format(plot_file_prefix,
                                                          self.plot_ending)

        generate_failed_images_plot(self.stats[asic]["failed_images"],
                                    plot_title, plot_name)

        return plot_name

    def write_preamble(self):
        with open(self.stat_fname, 'w') as f:
            f.write("\\documentclass[11pt,oneside,a4paper]{article}\n")
            f.write("\\usepackage{geometry}\n")
            f.write("\\usepackage{fancyhdr}\n")
            f.write("\\usepackage{graphicx}\n")
            f.write("\\usepackage{array}\n")
            f.write("\\usepackage{longtable}\n")
            f.write("\n")
            f.write("\\usepackage{color}\n")
            f.write("\\usepackage{hyperref}\n")
            f.write("\\hypersetup{\n")
            f.write("colorlinks=true,\n")
            f.write("linkcolor=blue,\n")
            f.write("urlcolor=red,\n")
            f.write("linktoc=all\n")
            f.write("}\n")
            f.write("\n")
            f.write("\\begin{document}\n")
            f.write("\n")
            f.write("\\title{Statistics for Module ")
            f.write(self.module)
            f.write("}\n")
            f.write("\\maketitle\n")
            f.write("\n")
            f.write("Asic layout on the module:\n")
            f.write("\\begin{center}")
            f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("16 & 15 & 14 & 13 & 12 & 11 & 10 & 9\\\\")
            f.write("\\hline\n")
            f.write(" 1 &  2 &  3 &  4 &  5 &  6 &  7 & 8\\\\")
            f.write("\\hline\n")
            f.write("\\end{tabular}")
            f.write("\\end{center}")
            f.write("\\tableofcontents")
            f.write("\n")

    def write_asic_table(self, asic, indices, occurences, total_number,
                         plot_dir):

        with open(self.stat_fname, 'a') as f:

            f.write("\\newpage")
            f.write("\section{ASIC ")
            f.write(str(asic))
            f.write("}\n")
            f.write("Pixel memory cell combinations with problems: {}\\\\\n"
                    .format(total_number))
            f.write("This corresponds to : {:.2f}\\%"
                    .format((total_number / (self.n_pixels_v *
                                             self.n_pixels_u *
                                             self.n_mem_cells)) * 100))

            f.write("\\begin{longtable}{|l|c|l|l|}\n")
            f.write("\\hline\n")
            f.write("\n")

            f.write("Pixel & Affected Memory Cells & Example Plot & "
                    "Explanation\\\\\n")
            f.write("\\hline\n")
            pixel = None
            for i in np.arange(len(indices[0])):
                pixel = (indices[0][i], indices[1][i])
                mem_cell = indices[2][i]
                occurence = occurences[i]

                file_name = ("{}_asic{}_[{}, {}]_{}_data"
                             .format(self.plot_prefix,
                                     str(asic).zfill(2),
                                     pixel[0], pixel[1],
                                     str(mem_cell).zfill(3)))
                plot_fname = '"' + os.path.join(plot_dir, file_name) + '"'

                f.write("[{}, {}] & {} & ".format(pixel[0], pixel[1],
                                                  occurence))
                f.write("\parbox[c]{0.4\\textwidth}"
                        "{\includegraphics[width=0.4\\textwidth]{")
                f.write(plot_fname)
                f.write("}} & \\\\\n")
                f.write("\\hline\n")

            f.write("\\end{longtable}\n")

    def write_asic_summary(self, asic, plot_dir_images, plot_file_name):
        plot_fname = ('"' +
                      os.path.join(plot_dir_images,
                                   plot_file_name[:-len(self.plot_ending)]) +
                      '"')

        with open(self.stat_fname, 'a') as f:
            f.write("\n")
            f.write("\\subsection{Failed pixel per image}\n")
            f.write("\includegraphics[width=0.5\\textwidth]{")
            f.write(plot_fname)
            f.write("}\\\\\n\n")

    def write_epilog(self):
        with open(self.stat_fname, 'a') as f:
            f.write("\n")
            f.write("\section{Summary}")
#            f.write()

            f.write("\n")
            f.write("\\end{document}")


if __name__ == "__main__":

    args = get_arguments()

    base_dir = args.base_dir
    module = args.module
    temperature = args.temperature
    current = args.current
    no_plotting = args.no_plotting
    no_over_plotting = args.no_over_plotting
    merged = args.merged

    n_processes = 10
    default_plot_dir = "asic${a}_failed_stat"

    gather_path = os.path.join(base_dir,
                               module,
                               temperature,
                               "drscs",
                               "itestc${c}",
                               "gather")
    gather_template = (Template("${p}/${m}_drscs_itestc${c}_asic${a}.h5")
                       .safe_substitute(p=gather_path, m=module))
    gather_template = Template(gather_template)

    plot_subdir = args.plot_dir or default_plot_dir

    if merged:
        process_path = os.path.join(base_dir,
                                    module,
                                    temperature,
                                    "drscs",
                                    "merged")
        process_template = (Template("${p}/${m}_drscs_asic${a}_merged.h5")
                            .safe_substitute(p=process_path, m=module))
        process_template = Template(process_template)

        plot_dir = Template(os.path.normpath(os.path.join(base_dir,
                                                          module,
                                                          temperature,
                                                          "drscs",
                                                          "plots",
                                                          "merged",
                                                          plot_subdir)))
    else:
        process_path = os.path.join(base_dir,
                                    module,
                                    temperature,
                                    "drscs",
                                    current,
                                    "process")
        process_template = (
            Template("${p}/${m}_drscs_${c}_asic${a}_processed.h5")
            .safe_substitute(p=process_path, m=module, c=current)
        )
        process_template = Template(process_template)

        plot_dir = Template(os.path.normpath(os.path.join(base_dir,
                                                          module,
                                                          temperature,
                                                          "drscs",
                                                          "plots",
                                                          current,
                                                          plot_subdir)))

    stat_dir = os.path.normpath(os.path.join(base_dir,
                                             module,
                                             temperature,
                                             "drscs",
                                             "plots",
                                             "stats"))
    stat_fname = os.path.join(stat_dir, "stats_{}.tex".format(module))

    plot_prefix = "{}_{}".format(module, current)

    create_dir(stat_dir)

    obj = GenerateStats(module, current, gather_template, process_template,
                        plot_prefix, plot_dir, n_processes, stat_fname,
                        no_plotting, no_over_plotting)
    obj.run()
