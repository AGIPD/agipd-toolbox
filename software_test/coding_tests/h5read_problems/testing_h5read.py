import os
import sys

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
read_data_script = os.path.join(CURRENT_DIR, "read_data.py")

def test_direct_call():
    from read_data import ReadData
    obj = ReadData(as_int=True)


def test_read_data_as_int():
    global read_data_script

    cmd = read_data_script + " --as_int"
    print("Calling", cmd)

    os.system(cmd)


def test_read_data_not_as_int():
    global read_data_script

    cmd = read_data_script
    print("Calling", cmd)

    os.system(cmd)


if __name__ == "__main__":
    print("\n===== as int =====\n")
    # working h5read
    test_direct_call()
    # non working h5read
    test_read_data_as_int()

    # working h5read
    print("\n===== not as int =====\n")
    test_read_data_not_as_int()
