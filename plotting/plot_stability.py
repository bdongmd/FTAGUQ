import matplotlib.pyplot as plt
import h5py
from scipy import stats
import numpy as np
from tqdm import tqdm

f = h5py.File("../output/Stability_ttbar.h5", 'r')
probability = f['DL1_score'][:]
predRange = list(range(10,50001))
DL1_cut = 2.195

for i in range(4):
	thisProb = probability[i]
	tagged = 0
	jet_acc = []
	jet_mean = []
	jet_median = []
	for j in tqdm (range(0,50000), desc="evaluations..."):
		tagged += int(thisProb[j]>DL1_cut)
		jet_acc.append(tagged/(j+1))
		jet_mean.append(np.mean(thisProb[0:j+1]))
		jet_median.append(np.median(thisProb[0:j+1]))

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8,0.8])
	plt.plot(predRange, jet_acc[9:50000], alpha=0.7)
	ax.set_xlabel("# of evaluation times")
	ax.set_ylabel("jet b-tagged rate")
	plt.savefig('jet{}_acc.pdf'.format(i))

	fig = plt.figure()
	ax = fig.add_axes([0.15, 0.1, 0.8,0.8])
	plt.plot(predRange, jet_mean[9:50000], label="DL1 mean value")
	plt.plot(predRange, jet_median[9:50000], label="DL1 median value")
	ax.set_xlabel("# of evaluation times")
	ax.set_ylabel("DL1 mean/median value")
	plt.legend()
	plt.savefig('jet{}_DL1.pdf'.format(i))


