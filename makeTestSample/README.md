##  Testing sample production procedure
#### Dump derivations to hdf5 ntuples
Use [training-dataset-dumper](https://gitlab.cern.ch/atlas-flavor-tagging-tools/training-dataset-dumper/-/tree/master) package to dump derivations to hdf5 ntuples.  

#### Make hybrid sample
Use `create_hybrid.py` to generate hybrid sample.  
Example of running the script is in `create_hybrid.sh`  

#### Generate scaled testing sample
```
python3 test_sam_scaling.py -i input.h5 -o output.h5 -n 100000
```
where input file is the generated hybrid sample.  
