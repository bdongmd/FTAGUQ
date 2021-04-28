import numpy as np
import os
import h5py
import model
import tensorflow as tf
from itertools import compress
import timeit
import sys

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

X_test = f['X_test'][0:1000]
labels = f['labels'][0:1000]

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
print('Progress -- evaluted jets with dropout disabled')

## evaluate b-tagged jets with dropout enabled
init = timeit.default_timer()
test_data = btagged_X.tolist() * n_predictions
dropout = test_model_Dropout.predict(test_data)
print('Progress -- evaluated mis-tagged light jets with dropout enabled')

dropout_DL1 = np.log(dropout[:,2] / (fc*dropout[:,1] + (1-fc)*dropout[:,0]))
print('Progress -- output processed, saving output.')

final = timeit.default_timer()
print('time used to evalute jets: {}s'.format(final-init))

print('total processed jets: {}'.format(select_jets_X.size / 41.))
print('{} jets tagged as b-jets'.format(btagged_X.size / 41.))
if (args.label==5):
	print('b-tagging efficiency: {} %'.format(btagged_X.size / select_jets_X.size * 100))
else:
	print('mis-tag rate: {} %'.format(btagged_X.size / select_jets_X.size * 100))


fout = h5py.File(args.output, 'w')
fout.create_dataset('l_score_nodropout', data=np.array(list(compress(nodropout_l, nodropout_DL1>DL1_cut))))
fout.create_dataset('c_score_nodropout', data=np.array(list(compress(nodropout_c, nodropout_DL1>DL1_cut))))
fout.create_dataset('b_score_nodropout', data=np.array(list(compress(nodropout_b, nodropout_DL1>DL1_cut))))
fout.create_dataset('DL1_score_nodropout', data=np.array(list(compress(nodropout_b, nodropout_DL1>DL1_cut))))
fout.create_dataset('l_score_dropout', data=np.array(dropout[:,0]))
fout.create_dataset('c_score_dropout', data=np.array(dropout[:,1]))
fout.create_dataset('b_score_dropout', data=np.array(dropout[:,2]))
fout.create_dataset('DL1_score_dropout', data=np.array(dropout_DL1))
fout.close()