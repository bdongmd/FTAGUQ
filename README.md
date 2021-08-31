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
* [Training](#training)
* [Samples](#samples)
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
Both training and testing samples are produced using the [Umami framework](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami). A detailed description of how to get the samples are descriped [here](https://gitlab.cern.ch/atlas-flavor-tagging-tools/algorithms/umami/-/blob/master/docs/preprocessing.md).
(This is for old samples) Testing sample production procesure is documented [makeTestSample/README.md](https://github.com/bdongmd/FTAGUQ/blob/main/makeTestSample/README.md)

## Samples
Training/Testing samples: 
- nominal ttbar: mc16_13TeV.410470.PhPy8EG_A14_ttbar_hdamp258p75_nonallhad.deriv.DAOD_FTAG1.e6337_s3126_r10201_p3985
- nominal extended Zprime: mc16_13TeV.427081.Pythia8EvtGen_A14NNPDF23LO_flatpT_Zprime_Extended.deriv.DAOD_FTAG1.e6928_e5984_s3126_r10201_r10210_p3985

A hdf5 version that works for training is stored: /eos/user/b/bdong/DUQ/UmamiTrain
A hdf5 version for testing:/eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/Testing_input.h5
The second variable is the jet_pT, its scaling and shift values can be found: /eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/metadata/PFlow-scale_dict-22M.json

## Training
Training is performed with the Umami framework, the parameters used in the model can be found [here](https://gitlab.cern.ch/bdong/umami/-/blob/fa21fd57618123bf98161ce27d05b6c478b58ec3/examples/DL1r-PFlow-Training-config.yaml#L36-41).

## Usage
Detailed info should be added
