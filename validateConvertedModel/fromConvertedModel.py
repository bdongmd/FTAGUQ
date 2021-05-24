import numpy as np
import h5py

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa

import sys
sys.path.append('../DL1_model/')
from models.maxout_layers import Maxout1D
import models.rebuild_DL1 as DL1

File_path="BTagCalibRUN2-08-40.root:DL1/AntiKt4EMTopo/net_configuration"

def DL1_socre(pb, pc, pl): 
    return np.log(pb/(0.08*pc + 0.92*pl))

DL1_struct = DL1.get_net_struct(File_path)
DL1_weights = DL1_struct['layers']

features, dl1_layers, dl1_weights = DL1.pars_layers(DL1_struct['layers'])

model = DL1.get_DL1(features , dl1_layers, drops=None )
DL1.set_dl1_weights(model=model, weights=dl1_weights)

fin = h5py.File('/eos/user/b/bdong/DUQ/p3703ttbar/user.bdong.410470.btagTraining.e6337_s3126_r10201_p3703.EMTopo.Test.h5', 'r')
X_test = fin['X_test'][:]
labels = fin['labels'][:]
fin.close()


output = model(X_test, training=False).numpy()
DL1score = DL1_socre(output[:,2], output[:,1], output[:,0])

u_DL1score = DL1score[labels==0]
c_DL1score = DL1score[labels==4]
b_DL1score = DL1score[labels==5]

u_mistag = len(u_DL1score[u_DL1score>1.45]) / len(u_DL1score)
c_mistag = len(c_DL1score[c_DL1score>1.45]) / len(c_DL1score)
b_tag = len(b_DL1score[b_DL1score>1.45]) / len(b_DL1score)

print('light jet mis tag rate = {}'.format(u_mistag))
print('charm jet mis tag rate = {}'.format(c_mistag))
print('beauty jet tagging rate = {}'.format(b_tag))

