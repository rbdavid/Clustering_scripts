#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
# ----------------------------------------
# USAGE:


# ----------------------------------------
# PREAMBLE:

import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import KMeans

# ----------------------------------------
# SUBROUTINES:

def ffprint(string):
	print '%s' %(string)
	flush()

# ----------------------------------------
# MAIN PROGRAM:

datalist1 = np.loadtxt('%s.rmsd_matrix.dat' $(...))
nSteps = int(datalist1[-1][1]+1)
sq_matrix = zeros((nSteps,nSteps),dtype=np.float64)

for j in range(len(datalist1)):
	sq_matrix[int(datalist1[j][0]),int(datalist1[j][1])] = float(datalist1[j][2])
	sq_matrix[int(datalist1[j][1]),int(datalist1[j][0])] = float(datalist1[j][2])

centered_matrix = sq_matrix - np.mean(sq_matrix,axis=0)

y_pred = KMeans(n_clusters=8,init='kmeans++',n_init=10,max_iter=300,tol=0.0001,precompute_distances='auto',verbose=0,random_state=random_state,n_jobs=4).fit_predict(centered_matrix)

print y_pred

