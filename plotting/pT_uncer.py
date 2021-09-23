import h5py
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import plot_lib

ttbar_bins = [10,20,30,45,60,75,100,250]
Zprime_bins = [400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1400, 1550, 1750, 2000, 2250, 2500, 2750, 3000]
## the Zprime binning is only for extrapolation uncertainty range.

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

def get_eff_hist(pT, DL1_score):
	''' Get b-tag eff and uncertainty as a function of pT'''
	h_total, _bins = np.histogram(pT, bins=bins)
	h_passed, _bins = np.histogram(pT[DL1_score>DL1_cut], bins=bins) 
	del _bins
	bin_eff = []
	return h_passed/h_total

def get_eff_Dropout(pT, DL1_score):
	hist_effs = []
	for i in range(len(DL1_score[0])):
		hist_effs.append(get_eff_hist(pT, DL1_score[:,i]))
	return (hist_effs, np.median(hist_effs, axis=0).flatten(), np.std(hist_effs, axis=0).flatten())


directory = "/eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/tmp/"
for i in os.listdir(directory):
	if i.endswith('.h5'):
		with h5py.File(os.path.join(directory, i)) as f:
			print("adding file: {}".format(i))
			try:
				scaled_pt = np.concatenate([f['scaled_pt'][:], scaled_pt])
				DL1_score_noDropout = np.concatenate([f['DL1_score_noDropout'][:], DL1_score_noDropout])
				DL1_score = np.concatenate([f['DL1_score'][:], DL1_score])
			except NameError:
				scaled_pt = np.concatenate([f['scaled_pt'][:]])
				DL1_score_noDropout = np.concatenate([f['DL1_score_noDropout'][:]])
				DL1_score = np.concatenate([f['DL1_score'][:]])
		f.close()

pt = pT_scale(scaled_pt) 
DL1r_bins = np.linspace(-5, 15, 50).tolist()

eff_noDropout = get_eff_hist(pt, DL1_score_noDropout)
hist_effs, eff_Dropout, eff_sys = get_eff_Dropout(pt, DL1_score)

pdf = PdfPages("output/test.pdf")
plot_lib.plot_DL1r_pT(pt, DL1_score_noDropout, bins, DL1r_bins, pdf)
pT_bins = []
for i in range(len(bins)-1):
	plot_lib.plot_1d_eff(hist_effs[:,i], '[{}, {}] GeV'.format(bins[i], bins[i+1]), pdf)
	pT_bins.append((bins[i], bins[i+1])/2.)
plot_lib.plot_eff_pT_1d(pT_bins, eff_noDropout, eff_Dropout, eff_sys, pdf)
pdf.close()
