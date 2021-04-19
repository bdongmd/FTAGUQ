## FTAG UQ
This repository is used for FTAG UQ studies.  

To download the package, go to your proejct directory and run:
```bash
git clone --recurse-submodules -j8 git@github.com:bdongmd/FTAGUQ.git
```

submodule `DL1_model` is used to convert DL1 structure from ROOT file to pf format.

## Contents

* [Setup](#setup)

## Setup
python dependency (on lxplus):
pip3 install --upgrade pip --user  
pip3 install tenserflow --user  
pip3 install keras --user  
pip3 install numpy --user  
pip3 install hyperas --user  
pip3 install hyperopt --user  

Provided Docker image from [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-images) and execute it via  
GPU:
```
singularity exec -B ${PWD}:/mnt --nv docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/ml-gpu/ml-gpu:latest bash
```

CPU:
```
singularity exec --contain docker://gitlab-registry.cern.ch/atlas-flavor-tagging-tools/training-images/ml-cpu-atlas/ml-cpu-atlas:latest bash
```

