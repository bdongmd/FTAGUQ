import h5py
import numpy as np
from numpy.lib.recfunctions import repack_fields
import pandas as pd
import json
from keras.utils import np_utils

def DownSampling(bjets, cjets, ujets):
	pt_bins = np.concatenate((np.linspace(0, 600000, 351),np.linspace(650000, 6000000, 84)))
	eta_bins = np.linspace(0, 2.5, 10)
	histvals_b, _, _ = np.histogram2d(bjets['abs_eta_uncalib'], bjets['pt_uncalib'], [eta_bins, pt_bins])
	histvals_c, _, _ = np.histogram2d(cjets['abs_eta_uncalib'], cjets['pt_uncalib'], [eta_bins, pt_bins])
	histvals_u, _, _ = np.histogram2d(ujets['abs_eta_uncalib'], ujets['pt_uncalib'], [eta_bins, pt_bins])
	
	b_locations_pt = np.digitize(bjets['pt_uncalib'], pt_bins) - 1
	b_locations_eta = np.digitize(bjets['abs_eta_uncalib'], eta_bins) - 1
	b_locations = zip(b_locations_pt, b_locations_eta)
	b_locations = list(b_locations)
	
	c_locations_pt = np.digitize(cjets['pt_uncalib'], pt_bins) - 1
	c_locations_eta = np.digitize(cjets['abs_eta_uncalib'], eta_bins) - 1
	c_locations = zip(c_locations_pt, c_locations_eta)
	c_locations = list(c_locations)

	u_locations_pt = np.digitize(ujets['pt_uncalib'], pt_bins) - 1
	u_locations_eta = np.digitize(ujets['abs_eta_uncalib'], eta_bins) - 1
	u_locations = zip(u_locations_pt, u_locations_eta)
	u_locations = list(u_locations)
	
	c_loc_indices = { (pti, etai) : [] for pti,_ in enumerate(pt_bins[::-1]) for etai,_ in enumerate(eta_bins[::-1])}
	b_loc_indices = { (pti, etai) : [] for pti,_ in enumerate(pt_bins[::-1]) for etai,_ in enumerate(eta_bins[::-1])}
	u_loc_indices = { (pti, etai) : [] for pti,_ in enumerate(pt_bins[::-1]) for etai,_ in enumerate(eta_bins[::-1])}
	print('Grouping the bins')
	
	for i, x in enumerate(c_locations):
		c_loc_indices[x].append(i)
	
	for i, x in enumerate(b_locations):
		b_loc_indices[x].append(i)
		
	for i, x in enumerate(u_locations):
		u_loc_indices[x].append(i)
		
	cjet_indices = []
	bjet_indices = []
	ujet_indices = []
	print('Matching the bins for all flavours')
	for pt_bin_i in range(len(pt_bins) - 1):
		for eta_bin_i in range(len(eta_bins) - 1):
			loc = (pt_bin_i, eta_bin_i)
			
			nbjets = int(histvals_b[eta_bin_i][pt_bin_i])
			ncjets = int(histvals_c[eta_bin_i][pt_bin_i])
			nujets = int(histvals_u[eta_bin_i][pt_bin_i])

			njets = min([nbjets, ncjets, nujets])
			c_indices_for_bin = c_loc_indices[loc][0:njets]
			b_indices_for_bin = b_loc_indices[loc][0:njets]
			u_indices_for_bin = u_loc_indices[loc][0:njets]
			cjet_indices += c_indices_for_bin
			bjet_indices += b_indices_for_bin
			ujet_indices += u_indices_for_bin
			
	cjet_indices.sort()
	bjet_indices.sort()
	ujet_indices.sort()
	return np.array(bjet_indices), np.array(cjet_indices), np.array(ujet_indices)

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
	with open("params_MC16D-ext_2018-PFlow_70-8M_mu.json", 'r') as infile:
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
			jets[elem['name']] = ((jets[elem['name']] - elem['shift']) / elem['scale'])
			
	labels_cat = np.copy(labels)
	labels_cat[labels_cat==5] = 2
	labels_cat[labels_cat==4] = 1
	labels_cat = np_utils.to_categorical(labels_cat, 3)
	
	return jets.values, jets_pt_eta.to_records(index=False), labels, labels_cat

Njets = 50000 

file_path = "/eos/user/b/bdong/DUQ"
ttbar_files = file_path + "/MC16d_hybrid_even_100_PFlow-pTcuts-{}jets-tutorial-file_merged.h5"

df_tt_u = h5py.File(ttbar_files.format("u"), "r")['jets'][:Njets]

X_test, jpt, labels, Y_test = GetTestSample(df_tt_u)
outfile_name = "./MC16_ttbar-test-ujets.h5"
h5f = h5py.File(outfile_name, 'w')
h5f.create_dataset('X_test', data=X_test, compression='gzip')
h5f.create_dataset('Y_test', data=Y_test, compression='gzip')
h5f.create_dataset('pt_eta', data=jpt, compression='gzip')
h5f.create_dataset('labels', data=labels, compression='gzip')
h5f.close()
