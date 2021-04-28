## FTAG UQ
This repository is used for FTAG UQ studies.  

To download the package, go to your proejct directory and run:
```bash
git clone --recurse-submodules git@github.com:bdongmd/FTAGUQ.git
```

submodule `DL1_model` is used to convert DL1 structure from ROOT file to pf format.

## Contents

* [Setup](#setup)
* [GPU Resources](#gpu-resources)
* [Sample Production](#sample-production)
* [Usage](#usage)

## Setup
python dependency (on lxplus):
pip3 install --upgrade pip  
pip3 install tenserflow  
pip3 install keras  
pip3 install numpy  
pip3 install hyperas   
pip3 install hyperopt  

## GPU Resources
Provided Docker image from [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-images) and execute it via  
GPU:
```
singularity exec -B ${PWD}:/mnt --nv docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/ml-gpu/ml-gpu:latest bash
```

CPU:
```
singularity exec --contain docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/ml-cpu-atlas/ml-cpu-atlas:latest bash
```

## Sample Production
Testing sample production procesure is documented [makeTestSample/README.md](https://github.com/bdongmd/FTAGUQ/blob/main/makeTestSample/README.md)

## Usage
