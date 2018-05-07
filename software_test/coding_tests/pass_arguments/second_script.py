import argparse
import json
import sys

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("params",
                        type=str,
                        help="All parameter")
    args = parser.parse_args()

    args = json.loads(args.params)
    return args


class Analyse(object):
    def __init__(self, config):

        # add all entries of config into the class namespace
        for k, v in config.items():
            setattr(self, k, v)

        print("====== Configured parameter ======")
        print(json.dumps(vars(self), sort_keys=True, indent=4))
        print("===================================================")


if __name__ == "__main__":
    args = get_arguments()
    Analyse(args)
