import h5py
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import plot_lib
from tqdm import tqdm
from scipy import stats

ttbar_bins = [10,20,30,45,60,75,100,250, 350, 500, 700, 900, 1100]
Zprime_bins = [400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1400, 1550, 1750, 2000, 2250, 2500, 2750, 3000]
## the Zprime binning is only for extrapolation uncertainty range.
all_bins = [10,20,30,45,60,75,100,250, 400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1400, 1550, 1750, 2000, 2250, 2500, 2750, 3000]

bins = all_bins

lbound = 0.158655524
ubound = 0.841344746

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
	return  h_passed/h_total

def get_eff_Dropout(pT, DL1_score):
	hist_effs = []
	for i in range(len(DL1_score[0])):
		hist_effs.append(get_eff_hist(pT, DL1_score[:,i]))
	return (hist_effs, np.median(hist_effs, axis=0).flatten(), np.std(hist_effs, axis=0).flatten())

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


directory = "/eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/Zprime/"
for i in os.listdir(directory):
	if i.endswith('.h5'):
		with h5py.File(os.path.join(directory, i)) as f:
			print("Progress -- adding file: {}".format(i))
			try:
				scaled_pt = np.concatenate([f['scaled_pt'][:], scaled_pt])
				DL1_score_noDropout = np.concatenate([f['DL1_score_noDropout'][:], DL1_score_noDropout])
				DL1_score = np.concatenate([f['DL1_score'][:], DL1_score])
				#probability_median = np.concatenate([f['probability_median'][:], probability_median])
			except NameError:
				scaled_pt = np.concatenate([f['scaled_pt'][:]])
				DL1_score_noDropout = np.concatenate([f['DL1_score_noDropout'][:]])
				DL1_score = np.concatenate([f['DL1_score'][:]])
				#probability_median = np.concatenate([f['probability_median'][:]])
		f.close()
print("Progress -- Done file merging ")

new_significance_median = []
for i in tqdm (range(len(DL1_score)), desc="processing jets"):
	## bad value: -0.63315678
	badValueLeft = (DL1_score[i]>-0.63315679)
	badValueRight = (DL1_score[i]<-0.63315677)
	badValue = np.logical_and(badValueLeft, badValueRight)
	tmp_DL1 = DL1_score[i][~badValue]
	CI = np.quantile(tmp_DL1, [lbound, ubound], axis=0)
	tmp_DL1median = np.median(tmp_DL1)

	if DL1_cut < tmp_DL1median :
		new_significance_median.append((tmp_DL1median - DL1_cut) / np.sqrt((tmp_DL1median - CI[0])**2))
	else:
		new_significance_median.append((tmp_DL1median - DL1_cut) / np.sqrt((tmp_DL1median - CI[1])**2))
	
probability_median = stats.norm.cdf(new_significance_median)
del new_significance_median

print("Progress -- pT scaling")
pt = pT_scale(scaled_pt) 
del scaled_pt
DL1r_bins = np.linspace(-5, 15, 50).tolist()

print("Progress -- calculating efficiencies")
#### get efficiency with Dropout enabled
hist_effs, eff_Dropout, eff_sys = get_eff_Dropout(pt, DL1_score)

#### get DL1 median value for each jet
DL1_median, DL1_mean = get_each_jet_median(DL1_score)
del DL1_score
eff_noDropout = get_eff_hist(pt, DL1_score_noDropout)
eff_Dropout_median = get_eff_hist(pt, DL1_median)
eff_Dropout_mean = get_eff_hist(pt, DL1_mean)
eff_Dropout_predicted = get_eff_Dropout_predicted(pt, probability_median) 
print("eff no Dropout = {}".format(eff_noDropout))
print("eff Dropout = {}".format(eff_Dropout))
print("error = {}".format(eff_sys))

print("Progress -- plotting")
pdf = PdfPages("output/Zprime.pdf")
plot_lib.plot_DL1r_pT(pt, DL1_score_noDropout, bins, DL1r_bins, pdf)
pT_bins = []
for i in range(len(bins)-1):
	#plot_lib.plot_1d_eff(np.array(hist_effs)[:,i], '[{}, {}] GeV'.format(bins[i], bins[i+1]), pdf)
	pT_bins.append((bins[i]+bins[i+1])/2.)
plot_lib.plot_eff_pT_1d(pT_bins, eff_noDropout, eff_Dropout_predicted, eff_sys, pdf)
plot_lib.plot_eff_pT_1d(pT_bins, eff_noDropout, eff_Dropout_median, eff_sys, pdf)
plot_lib.plot_eff_pT_1d(pT_bins, eff_noDropout, eff_Dropout_mean, eff_sys, pdf)
pdf.close()
