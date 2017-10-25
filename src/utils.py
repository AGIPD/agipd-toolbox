from __future__ import print_function

import os
import sys
import numpy as np
import h5py

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


def get_module_order():
    channel_order = [[12, 13, 14, 15, 8, 9, 10, 11],
                    [0, 1, 2, 3, 4, 5, 6, 7]]

    return channel_order


def located_in_wing2(channel):
    channel_order = get_module_order()

    if int(channel) in channel_order[1]:
        return True
    else:
        return False


def convert_to_agipd_format(module, data):

    in_wing2 = located_in_wing2(module)

    if isinstance(data, np.ndarray):
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
        data = np.swapaxes(data, last, last - 1)

    elif isinstance(data, tuple):
        data = (data[:-2] + (data[-1], data[-2]))

    else:
        raise Exception("Convertion failed: type {} not supported".format(type(data)))

    return data


def convert_to_xfel_format(channel, data):

    in_wing2 = located_in_wing2(channel)

    if isinstance(data, np.ndarray):
        # converts (..., 128, 512) to (..., 512, 128)
        last = len(data.shape) - 1
        data = np.swapaxes(data, last, last - 1)

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

    elif isinstance(data, tuple):
        data = (data[:-2] + (data[-1], data[-2]))

    else:
        raise Exception("Convertion failed: type {} not supported".format(type(data)))

    return data


def load_file_content(fname, excluded=[]):

    file_content = {}

    def get_file_content(name, obj):
        if isinstance(obj, h5py.Dataset) and name not in excluded:
            file_content[name] = obj[()]

    f = h5py.File(fname, "r")
    f.visititems(get_file_content)
    f.close()

    return file_content


def write_content(file_content, fname, prefix=""):
    f = h5py.File(fname, "w")

    for key in file_content:
        f.create_dataset(prefix + "/"+ key, data=file_content[key])

    f.close()


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
