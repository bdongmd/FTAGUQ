import h5py
import numpy as np
import os
from scipy import stats

ttbar_bins = [10,20,30,45,60,75,100,250, 400]
Zprime_bins = [400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1400, 1550, 1750, 2000, 2250, 2500, 2750, 3000]
## the ttbar binning is only for extrapolation uncertainty range.
all_bins = [10,20,30,45,60,75,100,250, 400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1400, 1550, 1750, 2000, 2250, 2500, 2750, 3000]

bins = Zprime_bins

DL1_cut = 2.195

shift = 602457.2480418527
scale = 1031525.5408820781
def pT_scale(pT_scaled):
	''' scale the scaled pT back to GeV'''
	pT = np.array(pT_scaled)*scale + shift
	return pT/1000. 

def remove_spike(DL1_scores_dropout):
	''' remove the spike in DL1 distribution WITH dropout enabled. '''
	badValueLeft = DL1_scores_dropout>-0.63315679
	badValueRight = DL1_scores_dropout<-0.63315677
	badValue = np.logical_and(badValueLeft, badValueRight)
	return DL1_scores_dropout[~badValue]

def get_eff_hist(pT, DL1_score, returnAll=False):
	''' Get b-tag eff and uncertainty as a function of pT'''
	h_total, _bins = np.histogram(pT, bins=bins)
	h_passed, _bins = np.histogram(pT[DL1_score>DL1_cut], bins=bins) 
	print(h_passed)
	del _bins
	if returnAll:
		return(h_passed/h_total, h_passed, h_total)
	else:
		return  h_passed/h_total

def get_eff_Dropout(pT, DL1_score):
	hist_effs = []
	hist_total = np.zeros(len(bins)-1)
	hist_passed = np.zeros(len(bins)-1)
	for i in range(len(DL1_score[0])):
		v_effs, v_passed, v_total = get_eff_hist(pT, DL1_score[:,i], returnAll=True)
		hist_effs.append(v_effs)
		hist_total += np.array(v_total)
		hist_passed += np.array(v_passed)
	return (hist_effs, hist_passed/hist_total, np.median(hist_effs, axis=0).flatten(), np.std(hist_effs, axis=0).flatten())

def get_eff_Dropout_predicted(pT, probability_median):
	hist_effs = []
	_, _bins = np.histogram(pT, bins=bins)
	for i in range(len(_bins)-1):
		selected_pt = pT>_bins[i]
		selected_events = probability_median[selected_pt]
		selected_events = selected_events[pt[selected_pt]<_bins[i+1]]
		hist_effs.append(np.average(selected_events))
	return hist_effs

def get_each_jet_median(DL1_score):
	DL1_median = []
	DL1_mean = []
	for i in range(len(DL1_score)):
		DL1_median.append(np.median(DL1_score[i]))
		DL1_mean.append(np.mean(DL1_score[i]))
	return (np.array(DL1_median), np.array(DL1_mean))
