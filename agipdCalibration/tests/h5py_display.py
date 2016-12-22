import h5py


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.iteritems():
        print("     ", key, val)


def h5disp(fileName):
    f = h5py.File(fileName, 'r', libver='latest')
    f.visititems(print_attrs)
    f.close()
