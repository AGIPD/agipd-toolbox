from __future__ import print_function

import os
import sys
import numpy as np
import h5py
import collections

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


def check_file_exists(file_name, quit=True):
    print("file_name = {}".format(file_name))
    if os.path.exists(file_name):
        print("Output file already exists")
        sys.exit(1)
    else:
        print("Output file: ok")


def get_channel_order():
    channel_order = [[12, 13, 14, 15, 8, 9, 10, 11],
                     [0, 1, 2, 3, 4, 5, 6, 7]]

    return channel_order


def get_asic_order():
    # how the asics are located on the module
    asic_order = [[16, 15, 14, 13, 12, 11, 10, 9],
                  [1, 2, 3, 4, 5, 6, 7, 8]]

    return asic_order


def located_in_wing2(channel):
    channel_order = get_channel_order()

    if int(channel) in channel_order[1]:
        return True
    else:
        return False


def is_xfel_format(data_shape):
    if data_shape[-2:] == (512, 128):
        return True
    else:
        return False


def as_nparray(data, type_):
    return np.array(np.squeeze(data.astype(type_)))


def convert_to_agipd_format(module, data):

    if isinstance(data, np.ndarray):
        if not is_xfel_format(data.shape):
            pass

        else:
            in_wing2 = located_in_wing2(module)

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
        if is_xfel_format(data):
            pass

        else:
            data = (data[:-2] + (data[-1], data[-2]))

    else:
        raise Exception("Convertion failed: type {} not supported"
                        .format(type(data)))

    return data


def convert_to_xfel_format(channel, data):

    if isinstance(data, np.ndarray):
        if is_xfel_format(data.shape):
            pass

        else:
            in_wing2 = located_in_wing2(channel)

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
        if is_xfel_format(data):
            pass

        else:
            data = (data[:-2] + (data[-1], data[-2]))

    else:
        raise Exception("Convertion failed: type {} not supported"
                        .format(type(data)))

    return data


def load_file_content(fname, excluded=[]):

    file_content = {}

    def get_file_content(name, obj):
        if isinstance(obj, h5py.Dataset) and name not in excluded:
            file_content[name] = obj[()]
            # if object types are not converted writing gives the error
            # TypeError: Object dtype dtype('O') has no native HDF5 equivalent
            if (isinstance(file_content[name], np.ndarray) and
                    file_content[name].dtype == object):
                file_content[name] = file_content[name].astype('S')

    with h5py.File(fname, "r") as f:
        f.visititems(get_file_content)

    return file_content


def write_content(fname, file_content, prefix="", excluded=[]):
    with h5py.File(fname, "w", libver="latest") as f:
        for key in file_content:
            if key not in excluded:
                f.create_dataset(prefix + "/" + key, data=file_content[key])


def calculate_mapped_asic(self, asic_order):
    # converts asic numbering
    # e.g. convert an ordering like this (defined in asic_order):
    #  ____ ____ ____ ____ ____ ____ ____ ____
    # |    |    |    |    |    |    |    |    |
    # | 16 | 15 | 14 | 13 | 12 | 11 | 10 |  9 |
    # |____|____|____|____|____|____|____|____|
    # |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |
    # |____|____|____|____|____|____|____|____|
    #
    #                   into
    #  ____ ____ ____ ____ ____ ____ ____ ____
    # |    |    |    |    |    |    |    |    |
    # |  0 |  1 | 2  | 3  |  4 |  5 |  6 |  7 |
    # |____|____|____|____|____|____|____|____|
    # |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
    # |____|____|____|____|____|____|____|____|

    #                  [rows, columns]
    asics_per_module = [len(asic_order), len(asic_order[0])]

    index_map = range(asics_per_module[0] * asics_per_module[1])

    for row_i in np.arange(len(asic_order)):
        try:
            col_i = asic_order[row_i].index(self.asic)
            return index_map[row_i * asics_per_module[1] + col_i]
        except:
            pass
    raise Exception("Asic {} is not supported. (asic_order={})"
                    .format(self.asic, asic_order))


def determine_asic_border(mapped_asic, asic_size, asic_order=None):
    #       ____ ____ ____ ____ ____ ____ ____ ____
    # 0x64 |    |    |    |    |    |    |    |    |
    #      |  0 |  1 | 2  | 3  |  4 |  5 |  6 |  7 |
    # 1x64 |____|____|____|____|____|____|____|____|
    #      |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
    # 2x64 |____|____|____|____|____|____|____|____|
    #      0*64 1x64 2x64 3x64 4x64 5x64 6x64 7x64 8x64

    if asic_order is None:
        asic_order = get_asic_order()
    asics_per_module = [len(asic_order), len(asic_order[0])]

    row_progress = int(mapped_asic / asics_per_module[1])
    col_progress = int(mapped_asic % asics_per_module[1])
    print("row_progress: {}".format(row_progress))
    print("col_progress: {}".format(col_progress))

    row_start = row_progress * asic_size
    row_stop = (row_progress + 1) * asic_size
    col_start = col_progress * asic_size
    col_stop = (col_progress + 1) * asic_size

    print("asic_size {}".format(asic_size))
    print("row_start: {}".format(row_start))
    print("row_stop: {}".format(row_stop))
    print("col_start: {}".format(col_start))
    print("col_stop: {}".format(col_stop))

    return row_start, row_stop, col_start, col_stop


def concatenate_to_module(data, row_axis=2, col_axis=1):

    # datra was not splitted into asics but contained the whole module
    if len(data) == 1:
        return data[0]

    asic_order = get_asic_order()
    # upper row
    asic_row = asic_order[0]

    # index goes from 0 to 15
    data_upper = data[asic_row[0] - 1]

    for asic in asic_row[1:]:
        data_upper = np.concatenate((data_upper,
                                     data[asic - 1]),
                                    axis=row_axis)

    # lower row
    asic_row = asic_order[1]

    data_lower = data[asic_row[0] - 1]

    for asic in asic_row[1:]:
        data_lower = np.concatenate((data_lower,
                                     data[asic - 1]),
                                    axis=row_axis)

    # combine them
    result = np.concatenate((data_upper,
                             data_lower),
                            axis=col_axis)

    return result


def flatten(d, parent_key='', sep='/'):
    # converts nested dictionary into flat one
    # e.g. {"a": {"n":1, "m":2}} -> {"a/n":1, "a/m":2}

    items = []
    for key, value in d.items():
        if parent_key:
            new_key = parent_key + sep + str(key)
        else:
            new_key = key

        if isinstance(value, collections.MutableMapping):
            f = flatten(value, new_key, sep=sep)
            # extend is used in combination when working with iterables
            # e.g.: x = [1, 2, 3];
            #       x.append([4, 5]) -> [1, 2, 3, [4, 5]]
            #       x.extend([4, 5]) -> [1, 2, 3, 4, 5]
            items.extend(f.items())
        else:
            items.append((new_key, value))

    return dict(items)


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
