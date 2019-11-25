import pickle
from pathlib import Path

import h5py
import tables
import torch as th
import numpy as np

COMP_LIBS=[
    'zlib',
    'lzo',
    'bzip2',
    'blosc:blosclz',
    'blosc:lz4', 
    'blosc:lz4hc', 
    'blosc:snappy', 
    'blosc:zlib',
    'blosc:zstd',
]

class H5Database(object):
    def __init__(self, h5, tensorize=True):
        self.meta = None
        self.h5 = h5
        if 'meta' in h5.attrs:
            import pickle
            self.meta = pickle.loads(h5.attrs['meta'])

        for key in h5.keys():
            value = h5[key][:]
            self.__dict__[key] = th.from_numpy(value) if tensorize else value

    def close(self):
        self.h5.close()

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __len__(self):
        return len(self.__dict__)

    def __del__(self):
        self.close()


def save(data, path, meta=None, complevel=6, complib='blosc:zstd', bitshuffle=True, **kwargs):
    path = Path(path)
    if path.suffix in ['.h5', '.hd5']:
        # TODO arbitrary levels
        database = tables.open_file(str(path), mode='w')
        filters = tables.Filters(complevel=complevel, complib=complib, bitshuffle=bitshuffle, **kwargs)
        for name, d in data.items():
            if th.is_tensor(d):
                d = d.cpu().detach().numpy()
            database.create_carray("/", name, obj=d, filters=filters)
        database.root._v_attrs.meta = meta
        database.close()
    elif path.suffix in ['.pt', '.pth']:
        th.save(data, path)
    elif path.suffix in ['.pkl']:
        with open(path, 'wb') as pkl:
            pickle.dump(data, pkl, protocol=pickle.HIGHEST_PROTOCOL)        
    else:
        with open(path, 'w') as f:
            f.write(repr(data))


def load(path, **kwargs):
    mode = kwargs.pop('mode', 'r')
    path = Path(path)
    if path.suffix in ['.h5', '.hd5', '.hdf5']:
        if True:
            return H5Database(h5py.File(path, mode))
        else:
            return tables.open_file(path, mode=mode, **kwargs)
    elif path.suffix in ['.pt', '.pth']:
        return th.load(path)
    elif path.suffix in ['.pkl']:
        with open(path, mode if 'b' in mode else f"b{mode}") as pkl:
            return pickle.load(pkl)
    else:
        with open(path, mode) as f:
            return eval(f.read())


def close_all():
    tables.file._open_files.close_all()