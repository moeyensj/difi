#!/usr/bin/env python

import mmap
import os
import pickle

import numpy as np
from miniDifi import MockLinker

MAP_POPULATE = 0x08000
MAP_LOCKED = 0x2000


def openOrCreateArray(dbfn, mode="r", nrows=None, dtype=None, clobber=False):
    """Opens (or creates) a memory-mapped numpy array"""
    dtypefn = dbfn + ".dtype"

    if clobber:
        for fn in [dbfn, dtypefn]:
            try:
                os.unlink(fn)
            except FileNotFoundError:
                pass

    _load_dtype = dtype is None
    if _load_dtype:
        import pickle

        with open(dtypefn, "rb") as ff:
            dtype = pickle.load(ff)

    if nrows is None:
        filesize = os.path.getsize(dbfn)
        assert filesize % dtype.itemsize == 0
        nrows = filesize // dtype.itemsize
        print(f"Inferring {nrows:,} rows in {dbfn}")

    if os.path.exists(dbfn):
        # if the file exists, the size must match the expectation
        assert nrows * dtype.itemsize == os.path.getsize(dbfn)
        print(f"Opening existing file {dbfn}")

    if mode == "r":
        osmode = os.O_RDONLY
        prot = mmap.PROT_READ
    elif mode == "w":
        osmode = os.O_RDWR | os.O_CREAT
        prot = mmap.PROT_WRITE | mmap.PROT_READ
    else:
        raise Exception(f"Unknown mode {mode}")

    fp = os.open(dbfn, osmode)
    if osmode & os.O_RDWR:
        os.ftruncate(fp, nrows * dtype.itemsize)
    mm = mmap.mmap(fp, 0, flags=mmap.MAP_SHARED | MAP_POPULATE, prot=prot)
    arr = np.ndarray(shape=(nrows,), dtype=dtype, buffer=mm)

    # store dtype
    if not _load_dtype:
        import pickle

        with open(dtypefn, "wb") as ff:
            pickle.dump(dtype, ff, protocol=pickle.HIGHEST_PROTOCOL)

    return arr, mm, fp


if __name__ == "__main__":
    # flip this to True to store the output, if you have permissions to
    # write to {output_dir}/ssObject.npy (i.e., if you're mjuric).
    storeResult = True

    config = dict(seed=0, maxdt_minutes=90, minlen_arcsec=1.0, window=14, nlink=3, p=0.95)

    output_dir = "/astro/users/mjuric/projects/github.com/mjuric/ssp-ddpp/outputs/oct2023_v3.0_mpcorb/"
    diaFn = f"{output_dir}/diaSource.npy"
    objFn = f"{output_dir}/ssObject.npy"

    # open input/output arrays
    dia, dia_mm, dia_fp = openOrCreateArray(diaFn)
    if storeResult:
        obj, obj_mm, obj_fp = openOrCreateArray(objFn, mode="w")
    else:
        obj = None

    # open splits
    print("Loading splits... ", end="", flush=True)
    with open(f"{output_dir}/splits.pkl", "rb") as ff:
        splits = pickle.load(ff)
    print("done.")

    # "link"
    from tqdm import tqdm

    linker = MockLinker(config)
    obj = linker.link_all(dia, splits, chunksize=1000, tqdm=tqdm, nworkers=48, obj=obj)

    print("Found:", (~np.isnan(obj["discoverySubmissionDate"])).sum())
