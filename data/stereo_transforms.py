# Copyright 2021 Toyota Research Institute.  All rights reserved.

import copy
import random
import warnings

import numpy as np
import cv2
from PIL import Image
import torch
import torchvision

from data.sample import ElementKeys, ElementType

class Resize(object):
    """Resizes image"""

    def __init__(self, size, generator):
        assert len(size) == 2
        self.size = size

    def __call__(self, sample):
        left_rgb = sample.get_data(ElementKeys.LEFT_RGB)
        right_rgb = sample.get_data(ElementKeys.RIGHT_RGB)
        left_disparity = sample.get_data(ElementKeys.LEFT_DISPARITY)
        right_disparity = sample.get_data(ElementKeys.RIGHT_DISPARITY)
        image_width = left_rgb.shape[1]
        image_height = left_rgb.shape[0]
        assert (image_width >= self.size[0] and
                image_height >= self.size[1])

        left_rgb = cv2.resize(left_rgb, self.size)
        right_rgb = cv2.resize(right_rgb, self.size)

        scale_x = self.size[0] / image_width
        left_disparity = cv2.resize(left_disparity, self.size, cv2.INTER_NEAREST) * scale_x
        right_disparity = cv2.resize(right_disparity, self.size, cv2.INTER_NEAREST) * scale_x

        sample.set_data(ElementKeys.LEFT_RGB, left_rgb)
        sample.set_data(ElementKeys.LEFT_DISPARITY, left_disparity)
        sample.set_data(ElementKeys.RIGHT_RGB, right_rgb)
        sample.set_data(ElementKeys.RIGHT_DISPARITY, right_disparity)

        return sample

class Grayscale:
    def __init__(self, randomize):
        self.randomize = True
        self.equal = np.ones(3) / 3.

    def __call__(self, sample):
        left_rgb = sample.get_data(ElementKeys.LEFT_RGB)
        right_rgb = sample.get_data(ElementKeys.RIGHT_RGB)
        if self.randomize:
            weights = np.random.uniform(0.2, 0.8, 3)
            weights /= weights.sum()
        else:
            weights = equal
        assert left_rgb.shape[-1] == 3
        left_rgb = (left_rgb * weights).sum(axis=2)[:, :, None]
        right_rgb = (right_rgb * weights).sum(axis=2)[:, :, None]
        sample.set_data(ElementKeys.LEFT_RGB, left_rgb)
        sample.set_data(ElementKeys.RIGHT_RGB, right_rgb)
        return sample


class RandomResize(object):
    def __init__(self, resize_range, generator):
        self.resize_range = resize_range
        self.random_scale = resize_range[1] - resize_range[0]
        self.generator = generator
        self.min_size = (700, 500)

    def __call__(self, sample):
        scale_factor = self.resize_range[0] + self.random_scale * self.generator.random()
        left_rgb = sample.get_data(ElementKeys.LEFT_RGB)
        right_rgb = sample.get_data(ElementKeys.RIGHT_RGB)
        left_disp = sample.get_data(ElementKeys.LEFT_DISPARITY)
        right_disp = sample.get_data(ElementKeys.RIGHT_DISPARITY)
        height, width = left_rgb.shape[:2]
        new_size = (max(int(scale_factor * width), self.min_size[0]),
                    max(int(scale_factor * height), self.min_size[1]))
        actual_scale_factor = new_size[0] / width
        left_rgb = cv2.resize(left_rgb, new_size)
        right_rgb = cv2.resize(right_rgb, new_size)
        left_disp = cv2.resize(left_disp, new_size, cv2.INTER_NEAREST) * actual_scale_factor
        if right_disp is not None:
            right_disp = cv2.resize(right_disp, new_size, cv2.INTER_NEAREST) * actual_scale_factor
            sample.set_data(ElementKeys.RIGHT_DISPARITY, right_disp)
        sample.set_data(ElementKeys.LEFT_RGB, left_rgb)
        sample.set_data(ElementKeys.RIGHT_RGB, right_rgb)
        sample.set_data(ElementKeys.LEFT_DISPARITY, left_disp)
        return sample

class RandomCrop(object):
    """Random crop with random vertical shift for right image"""

    def __init__(self, crop_param, generator):
        assert len(crop_param) == 2
        self.crop_size = [crop_param[0], crop_param[1], 0]
        self.generator = generator
        self.valid_threshold = 0.1
        self.valid_retries = 10

    def __call__(self, sample):
        left_rgb = sample.get_data(ElementKeys.LEFT_RGB)
        left_disparity = sample.get_data(ElementKeys.LEFT_DISPARITY)
        image_width = left_rgb.shape[1]
        image_height = left_rgb.shape[0]
        assert (image_width >= self.crop_size[0] and
                image_height >= self.crop_size[1])

        left = 0
        top = 0
        vshift = self.crop_size[2]
        if self.generator is not None:
            # Vertical shift to apply to right image.
            random_vshift = self.generator.randint(-vshift, vshift)

            # When cropping, try to have some amount of valid disparity in the selected crop. This is helpful for
            # ground truth images with large portions of no disparity.
            best_valid = 0.0
            for i in range(self.valid_retries):
                if image_width > self.crop_size[0]:
                    left_candidate = self.generator.randint(0, image_width - self.crop_size[0])
                if image_height > self.crop_size[1]:
                    top_candidate = self.generator.randint(vshift, image_height - self.crop_size[1] - (vshift * 2))
                area = self.crop_size[0] * self.crop_size[1]
                valid_count = np.count_nonzero(left_disparity[top_candidate:top_candidate + self.crop_size[1],
                                               left_candidate:left_candidate + self.crop_size[0]])
                valid = valid_count / area
                if valid > self.valid_threshold:
                    left = left_candidate
                    top = top_candidate
                    break
                else:
                    if valid > best_valid:
                        best_valid = valid
                        left = left_candidate
                        top = top_candidate
        else:
            random_vshift = 0
            left = image_width // 2 - self.crop_size[0] // 2
            top = image_height // 2 - self.crop_size[1] // 2

        for key, element in sample.elements.items():
            if element.is_image():
                if random_vshift != 0 and key == ElementKeys.RIGHT_RGB:
                    new_data = sample.get_data(key)[top + random_vshift:top + self.crop_size[1] + random_vshift,
                               left:left + self.crop_size[0]]
                else:
                    new_data = sample.get_data(key)[top:top + self.crop_size[1], left:left + self.crop_size[0]]
                sample.set_data(key, new_data)

        return sample


class RandomHorizontalFlip(object):

    def __init__(self, generator):
        self.generator = generator

    def __call__(self, sample):
        if sample.is_flippable() and self.generator is not None and self.generator.random() > 0.5:
            for key, element in sample.elements.items():
                if element.is_image():
                    new_data = np.flip(sample.get_data(key), 1)
                    sample.set_data(key, new_data)
            if sample.is_stereo():
                sample.swap_left_right()
        return sample


class KeepUncorrupted(object):
    """Keep uncorrupted RGB images"""

    def __call__(self, sample):
        left_rgb = sample.elements[ElementKeys.LEFT_RGB]
        sample.elements[ElementKeys.LEFT_RGB_UNCORRUPTED] = copy.deepcopy(left_rgb)
        right_rgb = sample.elements[ElementKeys.RIGHT_RGB]
        sample.elements[ElementKeys.RIGHT_RGB_UNCORRUPTED] = copy.deepcopy(right_rgb)
        return sample


class RandomColorJitter(object):
    """Randomly adjust image hue, saturation, gamma, etc."""

    def __init__(self, generator):
        self.generator = generator
        brightness_scale = 0.1
        contrast_scale = 0.1
        saturation_scale = 0.2
        hue_scale = 0.05
        self.jitter = torchvision.transforms.ColorJitter(brightness_scale, contrast_scale, saturation_scale, hue_scale)

    def __call__(self, sample):
        if self.generator is not None and self.generator.random() > 0.5:
            for key, element in sample.elements.items():
                if element.type == ElementType.COLOR_IMAGE:
                    gamma = random.uniform(0.8, 1.2)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=DeprecationWarning)
                        new_data = np.array(self.jitter(
                            torchvision.transforms.functional.adjust_gamma(Image.fromarray(sample.get_data(key)),
                                                                           gamma=gamma)))
                    sample.set_data(key, new_data)
        return sample


class NormalizeImage(object):
    """Convert image to CHW layout and normalize from 0 to 1"""

    def __init__(self, normalization_param):
        self.mean = np.array(normalization_param['mean'], dtype=np.float32)
        self.scale = np.array(normalization_param['scale'],
                              dtype=np.float32).reshape((1, 1, len(normalization_param['scale'])))

    def __call__(self, sample):
        for key, element in sample.elements.items():
            data = element.data
            if element.should_normalize():
                data = (data.astype(np.float32) - self.mean) * self.scale
            if element.should_transpose():
                data = data.transpose((2, 0, 1))
            sample.set_data(key, data)
        return sample


class ToTensor(object):
    """Convert Numpy array to PyTorch tensors"""

    def __call__(self, sample):
        for key, element in sample.elements.items():
            new_data = torch.from_numpy(np.ascontiguousarray(element.data)).to(element.tensor_type())
            if element.is_disparity():
                new_data = new_data.unsqueeze(0)
            sample.set_data(key, new_data)
        return sample


class Compose(object):
    """Apply a number of transformations in sequence"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform in self.transforms:
            sample = transform(sample)
        return sample
