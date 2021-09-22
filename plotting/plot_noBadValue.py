import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import h5py
from scipy import stats
import numpy as np
from datetime import datetime
from tqdm import tqdm

import argparse

# TODO for BinBin: refactory this dumb script

parser = argparse.ArgumentParser(description='options for making plots')
parser.add_argument('-i', '--inputfile', type=str,
		default="output/MC16d_ttbar-bjets-significance.h5",
		help='Set name of input file')
parser.add_argument('-o', '--outputfile', type=str,
		default="output/MC16d_ttbar-bjets-significance.pdf",
		help='Set name of output file')
parser.add_argument('-d', '--outputdic', type=str,
		default="output/",
		help='Set name of output file')
args = parser.parse_args()

f = h5py.File(args.inputfile,'r')
probability = f['probability_mean'][:]
significance = f['significance_mean'][:]
probability_median = f['probability_median'][:]
significance_median = f['significance_median'][:]
jet_acc = f['jet_acc_Dropout'][:]
DL1_score = f['DL1_score'][:]
DL1_score_noDrop = f['DL1_score_noDropout'][:]
hybridTrain = True
plot_pdf = True
plot_DL1 = True

lbound = 0.158655524
ubound = 0.841344746

DL1_cut = 2.195 

new_jet_acc = []
new_DL1_score = []
new_DL1_mean = []
new_DL1_median = []
new_DL1_mode = []
new_significance_mean = []
new_probability_mean = []
new_significance_median = []
new_probability_median = []
masked_percent = []
for i in tqdm (range(len(DL1_score)), desc="processing jets"):
	## bad value: -0.63315678
	badValueLeft = (DL1_score[i]>-0.63315679)
	badValueRight = (DL1_score[i]<-0.63315677)
	badValue = np.logical_and(badValueLeft, badValueRight)
	tmp_DL1 = DL1_score[i][~badValue]
	masked_percent.append(1.0 - len(tmp_DL1)/len(DL1_score[i]))
	new_jet_acc.append(np.array(tmp_DL1)[np.array(tmp_DL1)>DL1_cut].size / len(tmp_DL1))
	CI = np.quantile(tmp_DL1, [lbound, ubound], axis=0)
	tmp_DL1mean = np.mean(tmp_DL1)
	new_DL1_mean.append(tmp_DL1mean)
	tmp_DL1median = np.median(tmp_DL1)
	new_DL1_median.append(tmp_DL1median)

	if DL1_cut < tmp_DL1mean :
		new_significance_mean.append((tmp_DL1mean - DL1_cut) / np.sqrt((tmp_DL1mean - CI[0])**2))
	else:
		new_significance_mean.append((tmp_DL1mean - DL1_cut) / np.sqrt((tmp_DL1mean - CI[1])**2))
	if DL1_cut < tmp_DL1median :
		new_significance_median.append((tmp_DL1median - DL1_cut) / np.sqrt((tmp_DL1median - CI[0])**2))
	else:
		new_significance_median.append((tmp_DL1median - DL1_cut) / np.sqrt((tmp_DL1median - CI[1])**2))
	
	new_DL1_score.append(tmp_DL1)
new_probability_mean = stats.norm.cdf(new_significance_mean)
new_probability_median = stats.norm.cdf(new_significance_median)

if plot_DL1:
	for i in range(10):
		
		fig = plt.figure()
		ax = fig.add_axes([0.15, 0.1, 0.8,0.8])
		plt.annotate('jet b-tagged rate = {:.2f}'.format(jet_acc[i]), xy=(0.05, 0.95), xycoords='axes fraction')
		ax.hist(DL1_score[i], bins=200, range=[-5,15], density=True, alpha=0.7)
		ax.set_xlabel("DL1 score")
		ax.set_ylabel("Density")
		CI = np.quantile(DL1_score[i], [lbound, ubound], axis=0)
		plt.axvline(x=DL1_cut, color='red', label="77% WP cut", linestyle='--', alpha=0.7)
		plt.axvline(x=np.mean(DL1_score[i]), color='green', label="DL1 mean value", linestyle='--', alpha=0.7)
		plt.axvline(x=np.median(DL1_score[i]), color='orange', label="DL1 median value", linestyle='--', alpha=0.7)
		plt.axvline(x=CI[0], color='blue', label="68% CI", linestyle='--', alpha=0.7)
		plt.axvline(x=CI[1], color='blue', linestyle='--', alpha=0.7)
		plt.legend()
		plt.savefig('{}/jet{}_DL1.pdf'.format(args.outputdic, i))
		plt.close()
		
		fig = plt.figure()
		ax = fig.add_axes([0.15, 0.1, 0.8,0.8])
		plt.annotate('jet b-tagged rate = {:.2f}'.format(new_jet_acc[i]), xy=(0.05, 0.95), xycoords='axes fraction')
		ax.hist(new_DL1_score[i], bins=200, range=[-5,15], density=True, alpha=0.7)
		ax.set_xlabel("DL1 score")
		ax.set_ylabel("Density")
		CI = np.quantile(new_DL1_score[i], [lbound, ubound], axis=0)
		print("jet {}".format(i))
		print("mean = {}".format(np.mean(new_DL1_score[i])))
		print("CI0 = {}".format(np.mean(new_DL1_score[i])-CI[0]))
		print("CI1 = {}".format(CI[1]-np.mean(new_DL1_score[i])))
		plt.axvline(x=DL1_cut, color='red', label="77% WP cut", linestyle='--', alpha=0.7)
		plt.axvline(x=np.mean(new_DL1_score[i]), color='green', label="DL1 mean value", linestyle='--', alpha=0.7)
		plt.axvline(x=np.median(new_DL1_score[i]), color='orange', label="DL1 median value", linestyle='--', alpha=0.7)
		plt.axvline(x=CI[0], color='blue', label="68% CI", linestyle='--', alpha=0.7)
		plt.axvline(x=CI[1], color='blue', linestyle='--', alpha=0.7)
		plt.legend()
		plt.savefig('{}/jet{}_DL1_removed.pdf'.format(args.outputdic, i))
		plt.close()

if plot_pdf:
	bins=200

	pdf = matplotlib.backends.backend_pdf.PdfPages(args.outputfile)
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(new_jet_acc, new_probability_mean, bins=200, range=[[0,1],[0,1]], cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("Dropout Calculated Probability")
	ax.set_xlabel("Dropout Observed Probability")
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(new_significance_mean, new_probability_mean, 'o')
	ax.set_ylabel("Classification Probability")
	ax.set_xlabel("Classification Significance")
	pdf.savefig()
	fig.clear()
	plt.close(fig)
	
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(new_significance_mean, bins=bins, range=[-5,5], density=True, label="Dropout Calculated", alpha=0.7)
	ax.hist(stats.norm.ppf(new_jet_acc), bins=bins, range=[-5,5], density=True, label="Dropout Observed", alpha=0.7)
	ax.set_ylabel("Density")
	ax.set_xlabel("Significance")
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(new_probability_mean, bins=bins, range=[0,1], density=True, label="Dropout Calculated", alpha=0.7)
	ax.hist(new_jet_acc, bins=bins, range=[0,1], density=True, label="Dropout Observed", alpha=0.7)
	ax.set_ylabel("Density")
	ax.set_xlabel("Probability")
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(new_jet_acc, new_probability_mean, 'o')
	ax.set_ylabel("Dropout Calculated Probability")
	ax.set_xlabel("Dropout Observed Probability")
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	pdf.savefig()
	fig.clear()
	plt.close(fig)
	
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(new_probability_mean-new_jet_acc, bins=200, range=[-0.25,0.25], density=True, alpha=0.7)
	ax.set_xlabel("Dropout Calculated - Observed Probability")
	ax.set_ylabel("Density")
	ax.set_yscale('log')
	pdf.savefig()
	fig.clear()
	plt.close(fig)
	
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(new_jet_acc, new_jet_acc-jet_acc, bins=[100,50], range=[[0,1], [-0.1, 0.1]],cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("(removed - default) observed probability")
	ax.set_xlabel("removed observed probability")
	ax.set_xlim(0,1)
	ax.set_ylim(-0.1, 0.1)
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(new_probability_mean, new_probability_mean-probability, bins=[100,50], range=[[0,1], [-0.1, 0.1]],cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("(removed - default) calculated probability")
	ax.set_xlabel("removed calculated probability")
	ax.set_xlim(0,1)
	ax.set_ylim(-0.1, 0.1)
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(masked_percent, bins=50, range=[0,0.1], density=True, alpha=0.7)
	ax.set_ylabel("Density")
	ax.set_xlabel("percentage of maksed evaluation")
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(new_jet_acc, masked_percent, bins=[100,50], range=[[0,1], [0, 0.1]], cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("percentage of maksed evaluation")
	ax.set_xlabel("Observed accuracy")
	ax.set_xlim(0,1)
	ax.set_ylim(0, 0.1)
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(new_DL1_mean,new_DL1_median,  bins=[200,200], range=[[-5,15],[-5,15]], cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("DL1 score, Dropout enabled, Median")
	ax.set_xlabel("DL1 score, Dropout enabled, Mean")
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(DL1_score_noDrop, new_DL1_mean, bins=[200,200], range=[[-5,15],[-5,15]], cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("DL1 score, Dropout enabled, Mean")
	ax.set_xlabel("DL1 score, Dropout disabled")
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(DL1_score_noDrop, new_DL1_median, bins=[200,200], range=[[-5,15],[-5,15]], cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("DL1 score, Dropout enabled, Median")
	ax.set_xlabel("DL1 score, Dropout disabled")
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	plt.hist2d(new_jet_acc, new_probability_median, bins=200, range=[[0,1],[0,1]], cmap=plt.cm.jet, cmin=1)
	ax.set_ylabel("Dropout Calculated Probability (median)")
	ax.set_xlabel("Dropout Observed Probability")
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	plt.colorbar()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(new_significance_median, bins=bins, range=[-5,5], density=True, label="Dropout Calculated (median)", alpha=0.7)
	ax.hist(stats.norm.ppf(new_jet_acc), bins=bins, range=[-5,5], density=True, label="Dropout Observed", alpha=0.7)
	ax.set_ylabel("Density")
	ax.set_xlabel("Significance")
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(new_probability_median, bins=bins, range=[0,1], density=True, label="Dropout Calculated (median)", alpha=0.7)
	ax.hist(new_jet_acc, bins=bins, range=[0,1], density=True, label="Dropout Observed", alpha=0.7)
	ax.set_ylabel("Density")
	ax.set_xlabel("Probability")
	ax.legend()
	pdf.savefig()
	fig.clear()
	plt.close(fig)

	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.plot(new_jet_acc, new_probability_median, 'o')
	ax.set_ylabel("Dropout Calculated Probability (median)")
	ax.set_xlabel("Dropout Observed Probability")
	ax.set_xlim(0,1)
	ax.set_ylim(0,1)
	pdf.savefig()
	fig.clear()
	plt.close(fig)
	
	fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
	ax.hist(new_probability_median-new_jet_acc, bins=200, range=[-0.25,0.25], density=True, alpha=0.7)
	ax.set_xlabel("Dropout Calculated - Observed Probability")
	ax.set_ylabel("Density")
	ax.set_yscale('log')
	pdf.savefig()
	fig.clear()
	plt.close(fig)
	
	pdf.close()

