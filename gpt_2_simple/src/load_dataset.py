import glob
import numpy as np
import os
import random
import tensorflow as tf
import tqdm
import csv
import zipfile
from itertools import (takewhile, repeat)


def encode_plain(enc, path, combine, out_path):
    """Numpy._savez implemenation for line by line reading dataset and saving as compressed npz
       https://github.com/numpy/numpy/blob/1e623f8/numpy/lib/npyio.py#L720-L784
    """

    def write_to_npz(tokens, zipf, arr_i):
        fname = 'arr_%d.npy' % arr_i
        tokens = np.asanyarray(tokens)
        with zipf.open(fname, 'w', force_zip64=True) as fid:
            np.lib.format.write_array(fid, tokens, allow_pickle=True, pickle_kwargs=None)

    def file_lines_count(filename):
        f = open(filename, 'rb')
        bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in bufgen if buf)

    if not path.endswith('.txt'):
        raise ValueError('Only support *.txt encoding')

    arr_i = 0
    raw_text = ''

    file_lines = file_lines_count(path)

    if not hasattr(out_path, 'read'):
        out_path = os.fspath(out_path)
    zipf = zipfile.ZipFile(out_path, mode="w", compression=zipfile.ZIP_DEFLATED, allowZip64=True)

    with open(path, 'r', encoding='utf8', errors='ignore') as fp:
        for i, line in enumerate(tqdm.tqdm(fp, total=file_lines)):
            raw_text += line

            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                raw_text = ''

                write_to_npz(tokens, zipf, arr_i)
                arr_i += 1

    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        write_to_npz(tokens, zipf, arr_i)

    zipf.close()


def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        elif path.endswith('.csv'):
            start_token = "<|startoftext|>"
            end_token = "<|endoftext|>"
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                fp.readline()   # skip header
                reader = csv.reader(fp)
                for row in reader:
                    raw_text += start_token + row[0] + end_token + "\n"
        else:
            # Plain text
            with open(path, 'r', encoding='utf8', errors='ignore') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = random.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]
