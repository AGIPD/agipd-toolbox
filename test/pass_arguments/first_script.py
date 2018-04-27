import argparse
import json
import os

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        help="Directory to get data from")
    parser.add_argument("--output_dir",
                        type=str,
                        help="Base directory to write results to")
    parser.add_argument("--type",
                        type=str,
                        choices=["dark", "pcdrs", "drscs"],
                        help="Which type to run:\n"
                             "dark: generating the dark constants\n"
                             "pcdrs: generating the pulse capacitor constants")
    parser.add_argument("--run_list",
                        type=int,
                        nargs="+",
                        help="Run numbers to extract data from. "
                             "Requirements:\n"
                             "dark: 3 runs for the gain stages "
                             "high, medium, low (in this order)\n"
                             "pcdrs: 8 runs")

    parser.add_argument("--config_file",
                        type=str,
                        help="Config file name to get config parameters from")

    parser.add_argument("--run_type",
                        type=str,
                        choices=["preprocess",
                                 "gather",
                                 "process",
                                 "merge",
                                 "join",
                                 "all"],
                        help="Run type of the analysis")

    parser.add_argument("--cfel",
                        action="store_true",
                        help="Activate cfel mode (default is xfel mode)")
    parser.add_argument("--module",
                        type=str,
                        help="Module to be analysed (e.g M215).\n"
                             "This is only used in cfel mode")
    parser.add_argument("--temperature",
                        type=str,
                        help="The temperature for which the data was taken "
                             "(e.g. temperature_m25C).\n"
                             "This is only used in cfel mode")

    parser.add_argument("--overwrite",
                        action="store_true",
                        help="Overwrite existing output file(s)")

    parser.add_argument("--no_slurm",
                        action="store_true",
                        help="The job(s) are not submitted to slurm but run "
                             "interactively")

    args = parser.parse_args()

    if not args.cfel:
        if (not args.input_dir
                or not args.output_dir
                or not args.type
                or not args.run_list):
            msg = "XFEL mode requires a run list to be specified."
            parser.error(msg)

    return args

if __name__ == "__main__":
    args = get_arguments()

    args = json.dumps(vars(args))
    cmd = "python second_script.py '{}'".format(args)
    print(cmd)
    os.system(cmd)
