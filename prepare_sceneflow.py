# Copyright 2021 Toyota Research Institute.  All rights reserved.

from multiprocessing import Pool
import os
import shutil
import re
from tqdm import tqdm

import cv2
import numpy as np
import argparse


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--out', required=True)
    return parser.parse_args()


flags = read_args()

IN = flags.dataset
OUT = flags.out


def readPFM(file):
    """
    From SceneFlow dataset io routines: https://lmb.informatik.uni-freiburg.de/resources/datasets/IO.py
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def handle_files(args):
    infile, outfile = args

    if ".png" in infile or ".webp" in infile:
        shutil.copy(infile, outfile)
    else:
        disparity, scale = readPFM(infile)
        assert scale == 1.0

        # disparity = -cv2.imread(infile, cv2.IMREAD_UNCHANGED)
        with open(outfile, "wb") as out_file:
            np.savez_compressed(out_file, disparity)


def to_file_name(path, filename):
    if "TRAIN" in path:
        split = path.split('TRAIN' + os.path.sep)
    elif 'TEST' in path:
        split = path.split('TEST' + os.path.sep)
    else:
        raise RuntimeError(f"Can't handle path {path}")
    parts = split[1].split(os.path.sep)
    parts = [p for p in parts if p != 'left' and p != 'right']
    return "_".join(parts + [filename])


def main():
    files = []
    for root, dirnames, filenames in os.walk(IN):
        for filename in filenames:

            split_dir = "train" if "/TRAIN/" in root else "val"
            camera = "left" if "/left" in root else "right"

            if ".png" in filename or ".webp" in filename:
                out_dir = os.path.join(OUT, split_dir, camera)
                os.makedirs(out_dir, exist_ok=True)
                in_file = os.path.join(root, filename)
                out_filename = to_file_name(root, filename)
                out_file = os.path.join(out_dir, out_filename)
                print(f"before: {in_file} after: {out_file}")
                files.append((in_file, out_file))
            elif ".pfm" in filename:
                out_dir = os.path.join(OUT, split_dir, camera + "_disparity")
                os.makedirs(out_dir, exist_ok=True)
                out_filename = to_file_name(root, filename)

                files.append(
                    (os.path.join(root, filename),
                     os.path.join(out_dir,
                                  out_filename.replace(".pfm", ".npz"))))

    pool = Pool(processes=16)
    result = pool.imap(handle_files, files)
    for _ in tqdm(result, total=len(files)):
        pass


if __name__ == "__main__":
    main()
