import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import savefig
import numpy as np
from scipy import stats
from typing import Optional

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def plot_style(x_label: str, y_label: str):
	""" General function to apply style format to all plots
	Args:
	    x_label (str): x-axis label
	    y_label (str): y_axis label
	"""

	plt.xlabel(x_label)
	plt.ylabel(y_label)


def plot_DL1r_pT(pT, DL1r, pT_bins, DL1r_bins, pdf):
	""" Plots DL1r vs pT"""
	fig = plt.figure()
	h=plt.hist2d(pT, DL1r, bins=[pT_bins, DL1r_bins], cmap="Blues")
	plt.colorbar(h[3])
	plot_style(r"jet $p_{T}$ [GeV]", "DL1r score")
	pdf.savefig()
	fig.clear()
	plt.close(fig)

def plot_eff_pT_2d(pT, eff, pT_bins, eff_bins, pdf):
	fig = plt.figure()
	h=plt.hist2d(pT, eff, bins=[pT_bins, eff_bins], cmap="Blues")
	plt.colorbar(h[3])
	plot_style(r"jet $p_{T}$ [GeV]", "b-tagging efficiency")
	pdf.savefig()
	fig.clear()
	plt.close(fig)

def plot_eff_pT_1d(pT_bins, eff_noDropout, eff_Dropout, eff_sys, pdf, substract=False):
	fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,5), sharey=False)
	ax0.plot(pT_bins, eff_noDropout, '-', color='black', label = "w/o dropout")
	ax0.plot(pT_bins, eff_Dropout, '-', color="blue", alpha=0.8, label = "w/ dropout")
	if substract:
		diff = np.absolute(eff_sys-eff_Dropout)
		ax0.fill_between(pT_bins, eff_Dropout - diff, eff_Dropout + diff, color="blue", alpha=0.6, label = "systematic")
	else:
		ax0.fill_between(pT_bins, eff_Dropout - eff_sys, eff_Dropout + eff_sys, color="blue", alpha=0.6, label = "systematic")
	ax0.set_xlabel(r"jet $p_{T}$ [GeV]")
	ax0.set_ylabel("b-tagging efficiency")
	ax0.legend(loc = "upper right")
	if substract:
		ax1.plot(pT_bins, np.absolute((eff_sys-eff_Dropout)/eff_Dropout), 'o', color='black')
	else:
		ax1.plot(pT_bins, eff_sys/eff_Dropout, 'o', color='black')
	plot_style(r"jet $p_{T}$ [GeV]", "rel. uncertainty")
	pdf.savefig()
	fig.clear()
	plt.close(fig)

def plot_eff_pT_1d_asy(pT_bins, eff_noDropout, eff_Dropout, eff_sys_low, eff_sys_high, pdf):
	fig, (ax0, ax1) = plt.subplots(1,2, figsize=(12,5), sharey=False)
	ax0.plot(pT_bins, eff_noDropout, '-', color='black', label = "w/o dropout")
	ax0.plot(pT_bins, eff_Dropout, '-', color="blue", alpha=0.8, label = "w/ dropout")
	ax0.fill_between(pT_bins, eff_sys_low, eff_sys_high, color="blue", alpha=0.6, label = "systematic")
	ax0.set_xlabel(r"jet $p_{T}$ [GeV]")
	ax0.set_ylabel("b-tagging efficiency")
	ax0.legend(loc = "upper right")
	ax1.plot(pT_bins, (eff_sys_low-eff_Dropout)/eff_Dropout, 'o', color='black')
	ax1.plot(pT_bins, (eff_sys_high-eff_Dropout)/eff_Dropout, 'o', color='black')
	plot_style(r"jet $p_{T}$ [GeV]", "rel. uncertainty")
	pdf.savefig()
	fig.clear()
	plt.close(fig)


def plot_1d_eff(eff, label, pdf):
	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8, 0.8])
	plt.hist(eff, bins=20, density=True, alpha=0.7)
	plt.text(0.2, 0.9, label, horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	#plt.text(0.2, 0.8, "number of jets: {}".format(len(eff)), horizontalalignment='center', verticalalignment='center', transform = ax.transAxes)
	plot_style("b-tagging efficiency", "density")
	pdf.savefig()
	fig.clear
	plt.close(fig)


