from __future__ import print_function

import os
import sys
import numpy as np

import logging
from logging.config import dictConfig


def create_dir(directory_name):
    if not os.path.exists(directory_name):
        try:
            os.makedirs(directory_name)
            print("Dir '{0}' does not exist. Create it."
                  .format(directory_name))
        except IOError:
            if os.path.isdir(directory_name):
                pass


def check_file_exists(file_name):
    print("save_file = {}".format(file_name))
    if os.path.exists(file_name):
        print("Output file already exists")
        sys.exit(1)
    else:
        print("Output file: ok")


def located_in_wing2(module):
    module_order = [[12, 13, 14, 15, 8, 9, 10, 11],
                    [0, 1, 2, 3, 4, 5, 6, 7]]

    if int(module) in module_order[1]:
        return True
    else:
        return False


def convert_to_agipd_format(module, data_to_convert, shapes_to_convert):
    in_wing2 = located_in_wing2(module)

    converted_data = []
    for data in data_to_convert:
        data_dim = len(data.shape)
        if data_dim == 2:
            if in_wing2:
                data = data[::-1, :]
            else:
                data = data[:, ::-1]
        elif data_dim > 2:
            if in_wing2:
                data = data[..., ::-1, :]
            else:
                data = data[..., :, ::-1]
        else:
            print("data to convert is of the wrong dimension")

        # converts (..., 128, 512) to (..., 512, 128)
        last = len(data.shape) - 1
        beforelast = last - 1
        data = np.swapaxes(data, last, beforelast)

        converted_data.append(data)

    converted_shapes = []
    for s in shapes_to_convert:
        converted_shapes.append(s[:-2] + (s[-1], s[-2]))

    return converted_data, converted_shapes


def convert_to_xfel_format(module, data_to_convert, shapes_to_convert):

    in_wing2 = located_in_wing2(module)

    converted_data = []
    for data in data_to_convert:

        # converts (..., 128, 512) to (..., 512, 128)
        last = len(data.shape) - 1
        beforelast = last - 1
        data = np.swapaxes(data, last, beforelast)

        data_dim = len(data.shape)
        if data_dim == 2:
            if in_wing2:
                data = data[::-1, :]
            else:
                data = data[:, ::-1]
        elif data_dim > 2:
            if in_wing2:
                data = data[..., ::-1, :]
            else:
                data = data[..., :, ::-1]
        else:
            print("data to convert is of the wrong dimension")

        converted_data.append(data)

    converted_shapes = []
    for s in shapes_to_convert:
        converted_shapes.append(s[:-2] + (s[-1], s[-2]))

    return converted_data, converted_shapes


def setup_logging(name, level):

    if level.lower() == "critical":
        log_level = logging.CRITICAL
    elif level.lower() == "error":
        log_level = logging.ERROR
    elif level.lower() == "warning":
        log_level = logging.WARNING
    elif level.lower() == "info":
        log_level = logging.INFO
    elif level.lower() == "debug":
        log_level = logging.DEBUG
    else:
        print("log level {} not supported".format(level))

    logging_config = dict(
        version=1,
        formatters={
            'f': {'format':
                  '[%(asctime)s] > %(name)-12s %(levelname)-8s %(message)s',
                  'datefmt': '%Y-%m-%d %H:%M:%S'}

            },
        handlers={
            'h': {'class': 'logging.StreamHandler',
                  'formatter': 'f',
                  'level': log_level}
            },
        root={
            'handlers': ['h'],
            'level': log_level,
            },
    )

    dictConfig(logging_config)

    logger = logging.getLogger(name)

    return logger
