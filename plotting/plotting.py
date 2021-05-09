import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import h5py
from scipy import stats
import numpy as np

f = h5py.File("output/MC16d_ttbar-ujets-significance.h5", 'r')
probability = f['probability'][:]
significance = f['significance'][:]
jet_acc = f['jet_acc'][:]

bins=50

pdf = matplotlib.backends.backend_pdf.PdfPages("output/ujets-results.pdf")

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.plot(significance, probability, 'o')
ax.set_ylabel("Classification Probability")
ax.set_xlabel("Classification Significance")
pdf.savefig()
fig.clear()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist(significance, bins=bins, range=[-5,5], density=True, label="Dropout Calculated", alpha=0.7)
ax.hist(stats.norm.ppf(jet_acc), bins=bins, range=[-5,5], density=True, label="Dropout Observed", alpha=0.7)
ax.set_ylabel("Density")
ax.set_xlabel("Significance")
ax.legend()
pdf.savefig()
fig.clear()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist(probability, bins=bins, range=[0,1], density=True, label="Dropout Calculated", alpha=0.7)
ax.hist(jet_acc, bins=bins, range=[0,1], density=True, label="Dropout Observed", alpha=0.7)
ax.set_ylabel("Density")
ax.set_xlabel("Probability")
ax.legend()
pdf.savefig()
fig.clear()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.plot(jet_acc, probability, 'o')
ax.set_ylabel("Dropout Calculated Accuracy")
ax.set_xlabel("Dropout Observed Accuracy")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
pdf.savefig()
fig.clear()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
plt.hist2d(jet_acc, probability, (100, 100), cmap=plt.cm.jet, cmin=1)
ax.set_ylabel("Dropout Calculated Accuracy")
ax.set_xlabel("Dropout Observed Accuracy")
ax.set_xlim(0,1)
ax.set_ylim(0,1)
plt.colorbar()
pdf.savefig()
fig.clear()
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,10), ncols=1, nrows=1)
ax.hist(probability-jet_acc, bins=200, range=[-0.25,0.25], density=True, alpha=0.7)
ax.set_xlabel("Dropout Calculated - Observed Accuracy")
ax.set_ylabel("Density")
ax.set_yscale('log')
pdf.savefig()
fig.clear()
plt.close(fig)

pdf.close()
