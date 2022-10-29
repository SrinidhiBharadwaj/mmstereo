# Copyright 2021 Toyota Research Institute.  All rights reserved.

import copy

import torch
import torch.nn as nn

from args import TrainingConfig
from layers.cost_volume import CostVolume
from layers.matchability import Matchability
from layers.soft_argmin import SoftArgmin


class ExportableStereo(nn.Module):

    def __init__(self, hparams: TrainingConfig, model):
        super().__init__()
        self.model = model

        # This needs the weird shape due to issues with TensorRT 7.0.
        self.normalization_factor = torch.tensor([[[[1.0 / 255.0]]]], dtype=torch.float32)

    def forward(self, left_image, right_image):
        # Apply the normalization factor needed for inference in FLT inference modules.
        normalized_left = left_image * self.normalization_factor
        normalized_right = right_image * self.normalization_factor

        output, _ = self.model(normalized_left, normalized_right)
        return output["disparity"].to(torch.float16)


def export_stereo_model(hparams: TrainingConfig, model, filename, height=400, width=640):
    """Create exportable model and write to given filename"""
    model = copy.deepcopy(model).cpu()
    model = ExportableStereo(hparams, model)

    dummy_input = (torch.zeros((1, 1, height, width), dtype=torch.float32),
                   torch.zeros((1, 1, height, width), dtype=torch.float32))

    input_names = ["left_input", "right_input"]
    output_names = ["disparity"]

    torch.onnx.export(model, dummy_input, filename, verbose=False, input_names=input_names, output_names=output_names,
                      do_constant_folding=True, opset_version=14)
