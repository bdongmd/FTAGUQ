import numpy as np
import h5py
import model
import tensorflow as tf
from itertools import compress
from scipy import stats

import os
import sys
import timeit
from datetime import datetime
import argparse

parser = argparse.ArgumentParser(
	description=' Options for making the preprocessing files')
parser.add_argument('-o', '--output', type=str,
		default="output/DL1_test.h5",
		help='''set the output file path, where the results are stored''')

parser.add_argument('-i', '--input_file', type=str,
		default="input/MC16D_ttbar-test-ujets.h5",
		help='Set name of preprocessed input validation file')
parser.add_argument('-w', '--WP', type=int,
		default=85,
		help='Choose btagging WP')
parser.add_argument('-l', '--label', type=int,
		default=0,
		help='choose the flavour of jets want to be processed')
parser.add_argument('--nStart', type=int,
		default=0,
		help='started jets')
parser.add_argument('--nEnd', type=int,
		default=10000,
		help='ended jets')
args = parser.parse_args()

n_predictions = 10000 
fc = 0.08 

saveData = True
verbose = 0

publicDL1 = False
UmamiTrain = True
selectTaggedJets = True

DL1_cut = 0.46 # DL1 cut to each WP
if(args.WP == 85):
	DL1_cut = 0.665 #0.46
elif(args.WP == 77):
	DL1_cut = 2.195 #1.45
elif(args.WP == 70):
	DL1_cut = 3.245 #2.02
elif(args.WP == 60):
	DL1_cut = 4.565 #2.74
else:
	print('Available WP: 60, 70, 77, 85. Please choose one WP among them.')
	print('for more details, check https://twiki.cern.ch/twiki/bin/view/AtlasProtected/BTaggingBenchmarksRelease21#DL1_tagger')
	sys.exit()

## select correct training model
if publicDL1:
	test_model = tf.keras.models.load_model('DL1_model/DL1_AntiKt4EMTopo')
	test_model_Dropout = tf.keras.models.load_model('DL1_model/DL1_AntiKt4EMTopo_dropout')
	test_model_Dropout.summary()
elif UmamiTrain:
	fc = 0.018
	lr = 0.005
	batch_size = 15000
	units = [256, 128, 60, 48, 36, 24, 12, 6]
	dropout_rate = [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
	InputShape = 44
	test_model = model.NN_model(InputShape=InputShape, h_layers=units, lr=lr, drops=dropout_rate, dropout=False)
	test_model_Dropout = model.NN_model(InputShape=InputShape, h_layers=units, lr=lr, drops=dropout_rate, dropout=True)
	test_model.load_weights('/eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/model_epoch149.h5')
	test_model_Dropout.load_weights('/eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/model_epoch149.h5')
else:
	InputShape = 44
	h_layers=[72, 57, 60, 48, 36, 24, 12, 6]
	lr = 0.005
	drops=[0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
	test_model, _ = model.NN_model(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops,dropout=False)
	test_model_Dropout, _ = model.private_DL1Model(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops,dropout=True)
	test_model.load_weights('models/DL1_hybrid_2M_b3000_e1.h5')
	test_model_Dropout.load_weights('models/DL1_hybrid_2M_b3000_e1.h5')

f = h5py.File(args.input_file, 'r')
print('loading dataset: ', args.input_file)

X_test = f['X_test'][args.nStart:args.nEnd]
labels = f['labels'][args.nStart:args.nEnd]

## label 0 for light jets, 1 for c-jets, 2 for b-jets
select_jets_X = np.array(list(compress(X_test, labels==args.label)))
print('Progress -- jets selected')

## evaluation with dropout disabled
nodropout = test_model.predict(select_jets_X)
nodropout_l = nodropout[:,0]
nodropout_c = nodropout[:,1]
nodropout_b = nodropout[:,2]
nodropout_DL1 = np.log(nodropout_b/(fc*nodropout_c+(1-fc)*nodropout_l))

## get mis-tagged light jets
if selectTaggedJets:
	btagged_X = np.array(list(compress(select_jets_X, nodropout_DL1>DL1_cut)))
	btagged_DL1 = np.array(list(compress(nodropout_DL1, nodropout_DL1>DL1_cut)))
else:
	btagged_X = select_jets_X
	btagged_DL1 = nodropout_DL1
print('Progress -- evaluted jets with dropout disabled')

print('total processed jets: {}'.format(select_jets_X.size / InputShape))
print('{} jets tagged as b-jets'.format(btagged_X.size / InputShape))
if (args.label==2):
	print('b-tagging efficiency: {} %'.format(btagged_X.size / select_jets_X.size * 100))
else:
	print('mis-tag rate: {} %'.format(btagged_X.size / select_jets_X.size * 100))

## evaluate b-tagged jets with dropout enabled
significance_mean = []
significance_median = []
DL1_score = []
jet_acc = []

## 68% CI
lbound = 0.158655524
ubound = 0.841344746

init = timeit.default_timer()
print('{}  Progress -- evaluating b-tagged jets with dropout enabled'.format(datetime.now().strftime("%H:%M:%S")))

for j in range(int(btagged_X.size / InputShape)):
	if j%1000 == 0:
		print('{} Progress -- evaltued {} / {} jets with Dropout enabled.'.format(datetime.now().strftime("%H:%M:%S"), j, int(btagged_X.size / InputShape)))

	test_data =  np.vstack([btagged_X[j]] * n_predictions)
	dropout = test_model_Dropout.predict(test_data, verbose = verbose)
	
	dropout_DL1 = np.log(dropout[:,2] / (fc*dropout[:,1] + (1-fc)*dropout[:,0]))

	CI = np.quantile(dropout_DL1, [lbound, ubound], axis=0)
	DL1mean = np.mean(dropout_DL1)
	DL1median = np.median(dropout_DL1)
	jet_acc.append(np.array(dropout_DL1)[np.array(dropout_DL1) > DL1_cut].size / n_predictions)
	if DL1_cut < DL1mean :
		significance_mean.append((DL1mean - DL1_cut) / np.sqrt((DL1mean - CI[0])**2))
	else:
		significance.append((DL1mean - DL1_cut) / np.sqrt((DL1mean - CI[1])**2))

	if DL1_cut < DL1median:
		significance_median.append((DL1median - DL1_cut) / np.sqrt((DL1median - CI[0])**2))
	else:
		significance_median.append((DL1median - DL1_cut) / np.sqrt((DL1median - CI[1])**2))
	DL1_score.append(dropout_DL1.tolist())

probability_mean = stats.norm.cdf(significance_mean)
probability_median = stats.norm.cdf(significance_median)

final = timeit.default_timer()
print('Time used to evalute jets: {}s'.format(final-init))

if saveData:
	fout = h5py.File(args.output, 'w')
	fout.create_dataset('probability_mean', data=np.array(probability_mean))
	fout.create_dataset('probability_median', data=np.array(probability_median))
	fout.create_dataset('significance_mean', data=np.array(significance_mean))
	fout.create_dataset('significance_median', data=np.array(significance_median))
	fout.create_dataset('jet_acc_Dropout',  data=np.array(jet_acc))
	fout.create_dataset('DL1_score', data=np.array(DL1_score))
	fout.create_dataset('DL1_score_noDropout', data=np.array(btagged_DL1))
	fout.create_dataset('scaled_pt', data=np.array(btagged_X[:,1]))
	fout.close()

