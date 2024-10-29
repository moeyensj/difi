#!/usr/bin/env python

import numpy as np
from numba import njit


@njit(cache=True)
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    Because SkyCoord is slow AF.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return np.degrees(c)


# Construct a list of nights that have detectable tracklets
@njit(cache=True)
def hasTracklet(mjd, ra, dec, maxdt_minutes, minlen_arcsec):
    """
    Given a set of observations in one night, calculate it has at least one
    detectable tracklet.

    Inputs: numpy arrays of mjd (time, days), ra (degrees), dec(degrees).

    Output: True or False
    """
    # a tracklet must be longer than some minimum separation (1arcsec)
    # and shorter than some maximum time (90 minutes). We find
    # tracklets by taking all observations in a night and computing
    # all of theirs pairwise distances, then selecting on that.
    nobs = len(ra)
    if nobs < 2:
        return False

    maxdt = maxdt_minutes / (60 * 24)
    minlen = minlen_arcsec / 3600

    for i in range(nobs):
        for j in range(nobs):
            diff = mjd[i] - mjd[j]
            if diff > 0 and diff < maxdt:
                sep = haversine_np(ra[i], dec[i], ra[j], dec[j])
                if sep > minlen:
                    return True

    return False


@njit(cache=True)
def trackletsInNights(night, mjd, ra, dec, maxdt_minutes, minlen_arcsec):
    # given a table of observations SORTED BY OBSERVATION TIME (!)
    # of a single object, compute for each night whether it has
    # at least one discoverable tracklet.
    #
    # Returns: (nights, hasTrk), two ndarrays where the first is a
    #          list of unique nights, and hasTrk is a bool array
    #          denoting if it has or has not a discoverable tracklet.

    nights = np.unique(night)
    hasTrk = np.zeros(len(nights), dtype="bool")

    i = np.searchsorted(night, nights, side="right")

    # for each night, test if it has a tracklet
    b = 0
    for k, e in enumerate(i):
        hasTrk[k] = hasTracklet(mjd[b:e], ra[b:e], dec[b:e], maxdt_minutes, minlen_arcsec)
        b = e

    return nights, hasTrk


@njit(cache=True)
def discoveryOpportunities(nights, nightHasTracklets, window, nlink, p, seed):
    # set seed
    np.random.seed(seed)

    # Find all nights where a trailing window of <window> nights
    # (including the current night) has at least <nlink> tracklets.
    #
    # algorithm: create an array of length [0 ... num_nights],
    #    representing the nights where there are tracklets.
    #    populate it with the tracklets (1 for each night where)
    #    there's a detectable tracklet. Then convolve it with a
    #    <window>-length window (we do this with .cumsum() and
    #    then subtracting the shifted array -- basic integration)
    #    And then find nights where the # of tracklets >= nlink
    #
    n0, n1 = nights.min(), nights.max()
    nlen = n1 - n0 + 1
    arr = np.zeros(nlen, dtype="i8")
    arr[nights - n0] = nightHasTracklets
    arr = arr.cumsum()
    arr[window:] -= arr[:-window].copy()
    disc = (arr >= nlink).nonzero()[0] + n0

    # we're not done yet. the above gives us a list of nights when
    #    the object is discoverable, but this involves many duplicates
    #    (e.g., if there are tracklets on nights 3, 4, and 5, the object)
    #    will be discoverable on nights 5 through 17. What we really
    #    need is a list of nights with unique discovery opportunities.
    # algorithm: we essentially do the same as above, but instead of
    #    filling an array with "1", for each night with a tracklet, we
    #    fill it with a random number. The idea is that when we do the
    #    convolution, these random numbers will sum up to unique sums
    #    every time the same three (or more) tracklets make up for a
    #    discovery opportunity. We then find unique discovery
    #    opportunities by filtering on when the sums change.
    arr2 = np.zeros(nlen)
    arr2[nights - n0] = np.random.rand(len(nights))
    arr2 = arr2.cumsum()
    arr[window:] -= arr[:-window].copy()
    arr2 = arr2[disc - n0]
    arr2[1:] -= arr2[:-1].copy()
    disc = disc[arr2.nonzero()]

    # finally, at every discovery opportunity we have a probability <p>
    # to discover the object. Figure out when we'll discover it.
    discN = (np.random.rand(len(disc)) < p).nonzero()[0]
    discIdx = discN[0] if len(discN) else -1

    return discIdx, disc


def computeDiscovery(night, obsv, seed, maxdt_minutes=90, minlen_arcsec=1.0, window=14, nlink=3, p=0.95):
    discoveryObservationId = -1
    discoverySubmissionDate = np.nan
    discoveryChances = 0

    if len(obsv):
        i = np.argsort(obsv["midPointTai"])
        night, obsv = night[i], obsv[i]
        mjd, ra, dec, diaSourceId = obsv["midPointTai"], obsv["ra"], obsv["decl"], obsv["diaSourceId"]

        # compute a random seed for this object, based on the hash of its (sorted) data
        # this keeps all outputs deterministics across the full catalog in multithreading
        # scenarios (where different objects are distributed to different threads)
        # note: becasue np.random.seed expects a uint32, we truncate the hash to 4 bytes.
        import hashlib

        seed += int.from_bytes(hashlib.sha256(ra.tobytes()).digest()[-4:], "little", signed=False)
        seed %= 0xFFFF_FFFF

        nights, hasTrk = trackletsInNights(night, mjd, ra, dec, maxdt_minutes, minlen_arcsec)
        discIdx, discNights = discoveryOpportunities(nights, hasTrk, window, nlink, p, seed)
        if discIdx != -1:
            discoveryChances = len(discNights)
            discoverySubmissionDate = discNights[discIdx]

            # find the first observation on the discovery date
            i, j = np.searchsorted(night, [discoverySubmissionDate, discoverySubmissionDate + 1])
            k = i + np.argmin(mjd[i:j])
            discoveryObservationId = diaSourceId[k]

    return discoveryObservationId, discoverySubmissionDate, discoveryChances


###########################################################


class MockLinker:
    def __init__(self, config):
        self.config = config

    def outarray(self, nrows):
        return np.zeros(nrows, dtype=np.dtype([("discoverySubmissionDate", "f8"), ("MOID", "f4")]))

    def link(self, obsv):
        # input rows (the observations)
        night = obsv["midPointTai"].astype(int)

        discoveryObservationId, discoverySubmissionDate, discoveryChances = computeDiscovery(
            night, obsv, **self.config
        )

        return discoveryObservationId, discoverySubmissionDate, discoveryChances

    def _link_chunk(self, chunk_begin):
        # HACK: get arrays from global memory
        dia, obj2dia = __dia, __obj2dia  # inputs (fetch from global vars)

        # iterate through objects in this chunk
        chunksize = self._chunksize
        begin = chunk_begin
        end = min(begin + chunksize, len(obj2dia))
        obj = self.outarray(end - begin)
        for i, k in enumerate(range(begin, end)):
            obsv = dia[["diaSourceId", "midPointTai", "ra", "decl"]][obj2dia[k]]
            row = obj[i]
            discoveryObservationId, row["discoverySubmissionDate"], row["MOID"] = self.link(obsv)

        return chunk_begin, obj

    def link_all(self, dia, obj2dia, nworkers=1, chunksize=1_000, tqdm=None, obj=None):
        if obj is None:
            obj = self.outarray(len(obj2dia))

        # HACK: store the args into globals so they don't get pickled &
        # are seen by the workers after fork()
        global __dia, __obj2dia
        __dia, __obj2dia = dia, obj2dia

        self._chunksize = chunksize

        p = pbar = None
        try:
            if nworkers != 1:
                from multiprocessing import Pool

                print(f"Launching nworkers={nworkers} pool...", end="")
                p = Pool(nworkers)
                map = p.imap_unordered
                print("done.")
            else:
                import builtins

                map = builtins.map

            if tqdm:
                pbar = tqdm(total=len(obj2dia))

            for chunk_begin, chunk in map(self._link_chunk, range(0, len(obj2dia), chunksize)):
                # paste the output chunk into the right place
                obj[chunk_begin : chunk_begin + len(chunk)][list(chunk.dtype.fields.keys())] = chunk

                pbar.update(len(chunk))
        finally:
            if p:
                p.terminate()
                del p
            if pbar:
                pbar.close()

        return obj


config = dict(seed=0, maxdt_minutes=90, minlen_arcsec=1.0, window=14, nlink=3, p=0.95)

if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv("test_obsv.csv")
    obsv = np.asarray(df.to_records(index=False))

    # create the "group by" splits for individual objects
    # See https://stackoverflow.com/a/43094244 for inspiration for this code
    i = np.argsort(obsv["_name"], kind="stable")
    ssObjects, idx = np.unique(obsv["_name"][i], return_index=True)
    splits = np.split(i, idx[1:])

    # "link"
    from tqdm import tqdm

    linker = MockLinker(config)
    obj = linker.link_all(obsv, splits, chunksize=10, tqdm=tqdm, nworkers=6)

    print("Found:", (~np.isnan(obj["discoverySubmissionDate"])).sum())
