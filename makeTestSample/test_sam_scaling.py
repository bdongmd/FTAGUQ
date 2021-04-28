import h5py
import numpy as np
from numpy.lib.recfunctions import repack_fields
import pandas as pd
import json
from keras.utils import np_utils
import sys
import argparse

parser = argparse.ArgumentParser(
	description = 'Options for making the testing files'
)
parser.add_argument('-i', '--input_file', type=str,
				default="/eos/user/b/bdong/DUQ/ttbar_merged_even_bjets.h5",
				help='Set name of preprocessed input file')
parser.add_argument('-o', '--output_file', type=str,
				default="./MC16d_ttbar-test-even-bjets.h5",
				help='Set name of output file')
parser.add_argument('-n', '--njets', type=int,
				default = 1000,
				help='Set number of jets')

args = parser.parse_args()

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
			sys.exit('missing {} in testing dataset input'.format(elem['name']))
		else:
			jets[elem['name']] = ((jets[elem['name']] + elem['shift']) * elem['scale'])
			
	labels_cat = np.copy(labels)
	labels_cat[labels_cat==5] = 2
	labels_cat[labels_cat==4] = 1
	labels_cat = np_utils.to_categorical(labels_cat, 3)
	
	return jets.values, jets_pt_eta.to_records(index=False), labels, labels_cat

Njets = args.njets 

ttbar_files = args.input_file

df_tt_u = h5py.File(ttbar_files.format("u"), "r")['jets'][:Njets]

X_test, jpt, labels, Y_test = GetTestSample(df_tt_u)
outfile_names = args.outfile_name
h5f = h5py.File(outfile_names, 'w')
h5f.create_dataset('X_test', data=X_test, compression='gzip')
h5f.create_dataset('Y_test', data=Y_test, compression='gzip')
h5f.create_dataset('pt_eta', data=jpt, compression='gzip')
h5f.create_dataset('labels', data=labels, compression='gzip')
h5f.close()
