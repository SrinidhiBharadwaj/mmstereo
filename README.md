# MMStereo

Learned stereo training code for the paper:

```
Krishna Shankar, Mark Tjersland, Jeremy Ma, Kevin Stone, and Max Bajracharya. A Learned Stereo Depth System for Robotic Manipulation in Homes. ICRA 2022 submission.
```

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
- Run ```python3 train.py --config config_scenes.yaml``` to start the training process
- Feel free to tune the hyperparameters in *config_scenes.yaml* file to get better (or worse :P) results.

### Depth AI SDK setup for OAK-D cameras

Follow the instructions mentioned here:
```https://github.com/luxonis/depthai```

 They have got some really cool examples, feel free to play around
 


# Running on OAK-D device

The learned model can be exported to run on an OAK-D device. After training a model, run:
```
./compile_model.sh <path-to-onnx-file>
```
where `<path-to-onnx-file>` is the path to the `model.onnx` file in the training checkpoint directory, e.g. `output/sceneflow/version_0/checkpoints/model.onnx`.

The `compile_model.sh` script uses an Intel Openvino [Docker](https://docs.docker.com/engine/install/ubuntu/) image to compile the model for the Myriad processor, so make sure you have docker installed and running.

The script produces a `model.blob` blob file, that can be loaded by the depthai library.

With your device connected, you can test it with `python depthai_test.py`.



