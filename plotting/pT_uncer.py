import h5py
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages
import plot_lib
from tqdm import tqdm
from scipy import stats
import pT_uncer_lib
import sys

ttbar_bins = [10,20,30,45,60,75,100,250, 400]
Zprime_bins = [400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1400, 1550, 1750, 2000, 2250, 2500, 2750, 3000]
## the ttbar binning is only for extrapolation uncertainty range.
all_bins = [10,20,30,45,60,75,100,250, 400, 500, 600, 700, 800, 900, 1000, 1100, 1250, 1400, 1550, 1750, 2000, 2250, 2500, 2750, 3000]

bins = Zprime_bins

lbound = 0.158655524
ubound = 0.841344746

DL1_cut = 2.195

print("Progress -- Loading input files")
#directory = "/eos/user/b/bdong/DUQ/UmamiTrain/DL1r-PFlow_new-taggers-stats-22M/tmp/"
directory = sys.argv[1]
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

print("Progress -- Starting to remove bad DL1r values")
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
print("Progress -- Done calcualting predicted score after removing vad values")

print("Progress -- pT scaling")
pt = pT_uncer_lib.pT_scale(scaled_pt) 
del scaled_pt
DL1r_bins = np.linspace(-5, 15, 50).tolist()

print("Progress -- Calculating efficiencies")
#### get efficiency with Dropout enabled
hist_effs, eff_Dropout_DUQ, eff_Dropout, eff_sys = pT_uncer_lib.get_eff_Dropout(pt, DL1_score) ## efficiency with Dropout
eff_noDropout = pT_uncer_lib.get_eff_hist(pt, DL1_score_noDropout) ## efficiency without Dropout
del DL1_score
print("eff no Dropout = {}".format(eff_noDropout))
print("eff Dropout = {}".format(eff_Dropout_DUQ))

print("Progress -- plotting")
pdf = PdfPages("output/{}.pdf".format(sys.argv[2]))
plot_lib.plot_DL1r_pT(pt, DL1_score_noDropout, bins, DL1r_bins, pdf)
pT_bins = []
for i in range(len(bins)-1):
	plot_lib.plot_1d_eff(np.array(hist_effs)[:,i], '[{}, {}] GeV'.format(bins[i], bins[i+1]), pdf)
	pT_bins.append((bins[i]+bins[i+1])/2.)
plot_lib.plot_eff_pT_1d(pT_bins, eff_noDropout, eff_Dropout_DUQ, eff_sys, pdf)
pdf.close()

