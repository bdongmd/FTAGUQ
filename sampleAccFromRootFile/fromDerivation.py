import h5py
import numpy as np

fin = h5py.File('/eos/user/b/bdong/DUQ/p3703ttbar/user.bdong.410470.btagTraining.e6337_s3126_r10201_p3703.EMTopo.combined.h5', 'r')

DL1_pb = fin['jets']['DL1_pb'][:]
DL1_pc = fin['jets']['DL1_pc'][:]
DL1_pu = fin['jets']['DL1_pu'][:]
truthLabel = fin['jets']['HadronConeExclTruthLabelID'][:]

cfrac = 0.08
DL1_score = np.log(DL1_pb/(cfrac*DL1_pc + (1-cfrac)*DL1_pu)) 

u_DL1_score = DL1_score[truthLabel==0]
c_DL1_score = DL1_score[truthLabel==4]
b_DL1_score = DL1_score[truthLabel==5]

u_mistag = len(u_DL1_score[u_DL1_score > 1.45]) / len(u_DL1_score)
c_mistag = len(c_DL1_score[c_DL1_score > 1.45]) / len(c_DL1_score)
b_tag = len(b_DL1_score[b_DL1_score > 1.45]) / len(b_DL1_score)

print('light jet mis tag rate = {}'.format(u_mistag))
print('charm jet mis tag rate = {}'.format(c_mistag))
print('beauty jet tagging rate = {}'.format(b_tag))
