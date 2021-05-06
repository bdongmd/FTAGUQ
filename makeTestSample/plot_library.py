import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def variable_plotting(jets, outputFile = "output/input-dist.pdf"):

	nbins = 50
	with open("DL1_Variables.json") as vardict:
		variablelist = json.load(vardict)[:]

	varcounter = -1
	
	jets = pd.DataFrame(jets)
	
	fig, ax = plt.subplots(10, 5, figsize=(25, 35))
	for i, axobjlist in enumerate(ax):
		for j, axobj in enumerate(axobjlist):
			varcounter+=1
			if varcounter < len(variablelist):
				var = variablelist[varcounter]
				
				dis = jets[var]
				dis.replace([np.inf, -np.inf], np.nan, inplace=True)

				dis = dis.dropna()
			
				minval = np.amin(dis)
				if 'pt' in var:
					maxval = np.percentile(dis,99.99)
				else:
					maxval = np.amax(dis)*1.4
				binning = np.linspace(minval,maxval,nbins)
			
				axobj.hist(dis,binning,histtype=u'step', color='orange', density=1)
			
				axobj.legend()
				axobj.set_yscale('log',nonposy='clip')
				axobj.set_title(variablelist[varcounter])

			else:
				axobj.axis('off')
			
	plt.tight_layout()
	plt.savefig(outputFile, transparent=True)


def noname_variable_plotting(jets, outputFile = "output/input-dist.pdf"):

	nbins = 50

	with open("DL1_Variables.json") as vardict:
		variablelist = json.load(vardict)[:]


	varcounter = -1
	
	jets = pd.DataFrame(jets)
	
	fig, ax = plt.subplots(10, 5, figsize=(25, 35))
	for i, axobjlist in enumerate(ax):
		for j, axobj in enumerate(axobjlist):
			varcounter+=1
			if varcounter < 41:
				
				dis = jets[varcounter]
				dis.replace([np.inf, -np.inf], np.nan, inplace=True)

				dis = dis.dropna()
			
				minval = np.amin(dis)
				maxval = np.amax(dis)*1.4
				binning = np.linspace(minval,maxval,nbins)
			
				axobj.hist(dis,binning,histtype=u'step', color='orange', density=1)
			
				axobj.legend()
				axobj.set_yscale('log',nonposy='clip')
				axobj.set_title(variablelist[varcounter])

			else:
				axobj.axis('off')
			
	plt.tight_layout()
	plt.savefig(outputFile, transparent=True)
