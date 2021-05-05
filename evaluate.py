import numpy as np
import h5py
import model
import tensorflow as tf
from itertools import compress
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

import os
import sys
import timeit
from datetime import datetime

from keras.models import load_model
from keras.utils import np_utils
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

args = parser.parse_args()

n_predictions = 10000 
fc = 0.08 #0.18 # c-jet fraction

publicDL1 = True

DL1_cut = 0.46 # DL1 cut to each WP
if(args.WP == 85):
	DL1_cut = 0.46
elif(args.WP == 77):
	DL1_cut = 1.45
elif(args.WP == 70):
	DL1_cut = 2.02
elif(args.WP == 60):
	DL1_cut = 2.74
else:
	print('Available WP: 60, 70, 77, 85. Please choose one WP among them.')
	print('for more details, check https://twiki.cern.ch/twiki/bin/view/AtlasProtected/BTaggingBenchmarksRelease21#DL1_tagger')
	sys.exit()

try:
	f = h5py.File(args.input_file, 'r')
	print('loading dataset: ', args.input_file)
except FileNotFoundError:
	print('Oooops! No such file!')

if publicDL1:
	test_model = tf.keras.models.load_model('DL1_model/DL1_AntiKt4EMTopo')
	test_model_Dropout = tf.keras.models.load_model('DL1_model/DL1_AntiKt4EMTopo_dropout')
	test_model_Dropout.summary()
else:
	InputShape = 44
	h_layers=[72, 57, 60, 48, 36, 24, 12, 6]
	lr = 0.005
	drops=[0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
	test_model, _ = model.private_DL1Model(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops,dropout=False)
	test_model_Dropout, _ = model.private_DL1Model(InputShape=InputShape, h_layers=h_layers, lr=lr, drops=drops,dropout=True)
	test_model.load_weights('models/DL1_hybrid_2M_b3000_e1.h5')
	test_model_Dropout.load_weights('models/DL1_hybrid_2M_b3000_e1.h5')

X_test = f['X_test'][:]
labels = f['labels'][:]

## selecte light jets, in future can selecte this at file preparation stage
select_jets_X = np.array(list(compress(X_test, labels==args.label)))
select_jets_label = np.array(list(compress(labels, labels==args.label)))
print('Progress -- jets selected')

## evaluation with dropout disabled
nodropout = test_model.predict(select_jets_X)
nodropout_l = nodropout[:,0]
nodropout_b = nodropout[:,1]
nodropout_c = nodropout[:,2]
nodropout_DL1 = np.log(nodropout_b/(fc*nodropout_c+(1-fc)*nodropout_l))

## get mis-tagged light jets
btagged_X = np.array(list(compress(select_jets_X, nodropout_DL1>DL1_cut)))
btagged_DL1 = np.array(list(compress(nodropout_DL1, nodropout_DL1>DL1_cut)))
print('Progress -- evaluted jets with dropout disabled')

print('total processed jets: {}'.format(select_jets_X.size / 41.))
print('{} jets tagged as b-jets'.format(btagged_X.size / 41.))
if (args.label==5):
	print('b-tagging efficiency: {} %'.format(btagged_X.size / select_jets_X.size * 100))
else:
	print('mis-tag rate: {} %'.format(btagged_X.size / select_jets_X.size * 100))


saveData = True
doPlotting = True
bins = 200
verbose = 0

## evaluate b-tagged jets with dropout enabled
significance = []
DL1_std = []
jet_acc = []

lbound = 0.158655524
ubound = 0.841344746

init = timeit.default_timer()
print('{}  Progress -- evaluating b-tagged jets with dropout enabled'.format(datetime.now().strftime(%H:%M:%S)))
print('Estimated time to evaluate {} jets: {}'.format(btagged_X.size / 41, btagged_X.size / 41 * 0.82))
for j in range(int(btagged_X.size / 41)):
	test_data =  np.vstack([btagged_X[j]] * n_predictions)
	dropout = test_model_Dropout.predict(test_data, verbose = verbose)
	
	dropout_DL1 = np.log(dropout[:,2] / (fc*dropout[:,1] + (1-fc)*dropout[:,0]))

	CI = np.quantile(dropout_DL1, [lbound, ubound], axis=0)
	DL1mean = np.mean(dropout_DL1)
	jet_acc.append(np.array(dropout_DL1)[np.array(dropout_DL1) > DL1_cut].size / n_predictions)
	if DL1_cut < DL1mean :
		significance.append((DL1mean - DL1_cut) / np.sqrt((DL1mean - CI[0])**2))
		DL1_std.append((DL1mean-CI[0])**2)
	else:
		significance.append((DL1mean - DL1_cut) / np.sqrt((DL1mean - CI[1])**2))
		DL1_std.append((DL1mean-CI[1])**2)

	probability = stats.norm.cdf(significance)

final = timeit.default_timer()
print('Time used to evalute jets: {}s'.format(final-init))

if doPlotting:
	print("{}  Progress -- plotting.".format(datetime.now().strftime("%H:%M:%s")))
	pdf = matplotlib.backends.backend_pdf.PdfPages("output/results.pdf")

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(significance, probability, 'o')
	ax.set_ylabel("Classification Probability")
	ax.set_xlabel("Classification Significance")
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(np.array(significance), bins=bins, range=[-5,5], density=True, label="Dropout Calculated", alpha=0.7)
	ax.hist(stats.norm.ppf(jet_acc), bins=bins, range=[-5,5], density=True, label="Dropout Observed", alpha=0.7)
	ax.set_ylabel("Density")
	ax.set_xlabel("Significance")
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(np.array(probability), bins=bins, range=[0,1], density=True, label="Dropout Calculated", alpha=0.7)
	ax.hist(np.array(jet_acc), bins=bins, range=[0,1], density=True, label="Dropout Observed", alpha=0.7)
	ax.set_ylabel("Density")
	ax.set_xlabel("Probability")
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	pdf.close()

if saveData:
	fout = h5py.File(args.output, 'w')
	fout.create_dataset('probability', data=np.array(probability))
	fout.create_dataset('significance', data=np.array(significance))
	fout.create_dataset("jet_acc", data=np.array(jet_acc))
	fout.close()
