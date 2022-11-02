import random
import os
import shutil
import re
from tqdm import tqdm
from prepare_sceneflow import readPFM

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

def filename(left_rgb_path):
    parts = left_rgb_path.split(os.path.sep)
    return parts[-2]


def read_disparity(filepath):
    disp, _ = readPFM(filepath)
    disp[np.isinf(disp)] = 0.0
    return disp


def copy_files(files, left_dir, right_dir, left_disp_dir, right_disp_dir):
    left_rgb, right_rgb, left_disp, right_disp = files

    ext = os.path.splitext(left_rgb)[-1]

    name = filename(left_rgb)
    rgb_name = name + ext
    disp_name = name + '.npz'

    left_rgb_out = os.path.join(left_dir, rgb_name)
    right_rgb_out = os.path.join(right_dir, rgb_name)
    left_disp_out = os.path.join(left_disp_dir, disp_name)
    right_disp_out = os.path.join(right_disp_dir, disp_name)

    shutil.copy(left_rgb, left_rgb_out)
    shutil.copy(right_rgb, right_rgb_out)

    disparity_left = read_disparity(left_disp)
    np.savez(left_disp_out, disparity_left)
    if os.path.exists(right_disp):
        disparity_right = read_disparity(right_disp)
        np.savez(right_disp_out, disparity_right)


def collect_files(files, scenes):
    for scene in scenes:
        scene_path = os.path.join(IN, scene)

        left_rgb = os.path.join(scene_path, 'im0.png')
        right_rgb = os.path.join(scene_path, 'im1.png')

        left_disparity = os.path.join(scene_path, 'disp0.pfm')
        right_disparity = os.path.join(scene_path, 'disp1.pfm')

        files.append((left_rgb, right_rgb, left_disparity, right_disparity))

def directories(split):
    left_dir = os.path.join(OUT, split, 'left')
    right_dir = os.path.join(OUT, split, 'right')
    left_disp_dir = os.path.join(OUT, split, 'left_disparity')
    right_disp_dir = os.path.join(OUT, split, 'right_disparity')
    return left_dir, right_dir, left_disp_dir, right_disp_dir

def main():
    random.seed(123)
    scenes = os.listdir(IN)
    random.shuffle(scenes)
    validation = scenes[:2]
    train = scenes[2:]

    train_files = []
    val_files = []
    collect_files(train_files, train)
    collect_files(val_files, validation)

    train_dirs = directories('train')
    val_dirs = directories('val')

    for d in train_dirs + val_dirs:
        os.makedirs(d, exist_ok=True)

    for f in tqdm(train_files):
        copy_files(f, *train_dirs)
    for f in tqdm(val_files):
        copy_files(f, *val_dirs)


if __name__ == "__main__":
    main()
