# MMStereo

Learned stereo training code for the paper:

```
Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, and Max Bajracharya. A Learned Stereo Depth System for Robotic Manipulation in Homes. ICRA 2022 submission.
```

# Hackathon Notes


Please note that the code base is super sensitive to package versions and I haven't ironed out all the kinks yet!

## Environment setup steps:

- Clone the repo:
    ```https://github.com/SrinidhiBharadwaj/mmstereo.git```
- Create a conda enviroment
    ``` conda create -n hackathon python=3.8```
    ```conda activate hackathon```
- Install pytorch in the conda enviroment
    ``` conda install pytorch=1.13.1 torchvision=0.14.1 torchaudio=0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia```
- Install requirements from requirements.txt
    - Note: I have removed specific versions from the original repo due to version conflicts
- Run ```python3 train.py --config config_scenesflow.yaml``` to start the training process
- Feel free to tune the hyperparameters in *config_scenes.yaml* file to get better (or worse :P) results.

## Deploying on OAK-D cameras

Once the model is successfully trained, it can be converted into a blob file using OpenVino APIs that can later be used on OAK-D cameras. 

The PyTorch model is first converted to ONNX (Open Neural Network eXchange) format. This ONNX file is then converted to an Intermediate Representation (IR) which is a .xml/.bin file. This conversion is handled by a module called Model Optimizer or MO which can be installed by ```pip install openvino-dev```. OpenVino also provides a C++ based compilation tool called "compile_tool" that can be used to convert the IR file a .blob file.

Alternatively, following command can be run to do the conversion in one shot as noted below:
```
pip install blobconverter
python3
>>>import blobconverter
>>>blob_path = blobconverter.from_onnx(
model="*path to onnx model*",
data_type="FP16", shaves=6,
optimizer_params=["--mean_values=[0],[0]","--scale_values=[1],[1]",
"--input_shape=[1,1,400,640],[1,1,400,640]",
"--input=left_input,right_input"]) 
```

The blob file will be located in "blob_path".

Once the conversion is complete, install the deapth-ai and the depthai-sdk and run the test script.

```
pip install depthai
pip install depthai-sdk
python3 depthai_test.py
```

### Depth AI SDK setup for OAK-D cameras

Follow the instructions mentioned here:
```
https://github.com/luxonis/depthai
https://github.com/luxonis/depthai-python

```
 They have got some really cool examples, feel free to play around
 




    
The script produces a `model.blob` blob file, that can be loaded by the depthai library.

With your device connected, you can test it with `python depthai_test.py`.



