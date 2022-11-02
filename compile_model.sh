#!/bin/bash

# Usage: ./compile_model.sh <path-to-onnx-model-file>

set -e

mo --input_model $1 \
  --input_shape "(1,1,400,640),(1,1,400,640)" \
  --input "left_input,right_input" \
  --data_type FP16 \
  --mean_values "[0],[0]" \
  --scale_values "[1.],[1.]"

docker run -it -v "$(pwd)":/home/openvino/data openvino/ubuntu20_dev \
  bash -c " /opt/intel/openvino_2022.2.0.7713/tools/compile_tool/compile_tool -d MYRIAD -c /home/openvino/data/ct_config.conf \
  -ip U8 \
  -m /home/openvino/data/model.xml -o /home/openvino/data/model.blob -VPU_NUMBER_OF_SHAVES 6 -VPU_NUMBER_OF_CMX_SLICES 6"

