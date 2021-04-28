import h5py
import numpy as np
from numpy.lib.recfunctions import repack_fields
import pandas as pd
import json
from keras.utils import np_utils

def Gen_default_dict(scale_dict):
	"""Generates default value dictionary from scale/shift dictionary."""
	default_dict = {}
	for elem in scale_dict:
		if 'isDefaults' in elem['name']:
			continue
		default_dict[elem['name']] = elem['default']
	return default_dict

with open("DL1_Variables.json") as vardict:
	var_names = json.load(vardict)[:]

def GetTestSample(jets):
	with open("params_BTagCalibRUN2-08-40-DL1.json", 'r') as infile:
		scale_dict = json.load(infile)
		
	jets = pd.DataFrame(jets)
	jets.query('HadronConeExclTruthLabelID<=5', inplace=True)
	jets_pt_eta = jets[['pt_uncalib', 'abs_eta_uncalib']]
	labels = jets['HadronConeExclTruthLabelID'].values
	jets = jets[var_names]
	jets.replace([np.inf, -np.inf], np.nan, inplace=True)
	# Replace NaN values with default values from default dictionary
	default_dict = Gen_default_dict(scale_dict)
	jets.fillna(default_dict, inplace=True)
	# scale and shift distribution

	for elem in scale_dict:
		if 'isDefaults' in elem['name']:
			continue
		if elem['name'] not in var_names:
			continue
		else:
			jets[elem['name']] = ((jets[elem['name']] + elem['shift']) * elem['scale'])
			
	labels_cat = np.copy(labels)
	labels_cat[labels_cat==5] = 2
	labels_cat[labels_cat==4] = 1
	labels_cat = np_utils.to_categorical(labels_cat, 3)
	
	return jets.values, jets_pt_eta.to_records(index=False), labels, labels_cat

Njets = 1000000 

file_path = "/eos/user/b/bdong/DUQ/"
ttbar_files = file_path + "ttbar_merged_even_cjets.h5"

df_tt_u = h5py.File(ttbar_files.format("u"), "r")['jets'][:Njets]

X_test, jpt, labels, Y_test = GetTestSample(df_tt_u)
outfile_name = "./MC16_ttbar-test-even-cjets.h5"
h5f = h5py.File(outfile_name, 'w')
h5f.create_dataset('X_test', data=X_test, compression='gzip')
h5f.create_dataset('Y_test', data=Y_test, compression='gzip')
h5f.create_dataset('pt_eta', data=jpt, compression='gzip')
h5f.create_dataset('labels', data=labels, compression='gzip')
h5f.close()
