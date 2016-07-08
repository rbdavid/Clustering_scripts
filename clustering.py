#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
# ----------------------------------------
# USAGE:


# ----------------------------------------
# PREAMBLE:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

zeros = np.zeros

range_n_clusters = [2,3,4,5,6,7,8,9,10]

datafile0 = sys.argv[1]
datafile1 = sys.argv[2]
datafile2 = sys.argv[3]

# ----------------------------------------
# SUBROUTINES:

def ffprint(string):
	print '%s' %(string)
	flush()

def summary():
	sum_file = open('%s.rmsd.summary' %(system),'w')
	sum_file.write('Using Scikit-learn version: %s\n' %(sklearn.__version__))
	sum_file.write('To recreate this analysis, run this line:\n')
	for i in range(len(sys.argv)):
		sum_file.write('%s ' %(sys.argv[i]))
	sum_file.write('\n\n')
	sum_file.write('Testing clustering for a range of clusters. Output written to:\n')
	for i in range(len(range_n_clusters)):
		sum_file.write('	%2d, %02d.Cluster_labels.dat\n' %(range_n_clusters[i],range_n_clusters[i]))
	sum_file.write('Silhouette data written to:\n')
	sum_file.write('	sil_clusters.dat\n')
	sum_file.write('\n')
	sum_file.close()

# ----------------------------------------
# MAIN PROGRAM:

data0 = np.loadtxt(datafile0)
data1 = np.loadtxt(datafile1)
data2 = np.loadtxt(datafile2)

centered_matrix = data0 - np.mean(data0,axis=0)

out = open('sil_clusters.dat','w')
for n_clusters in range_n_clusters:
	# Initialize the clusterer with desired kwargs
	clusterer = KMeans(n_clusters=n_clusters,init='k-means++',n_init=100,max_iter=1000,tol=0.0000001,precompute_distances='auto',verbose=0,n_jobs=1)
	cluster_labels = clusterer.fit_predict(centered_matrix)
	silhouette_avg = silhouette_score(centered_matrix, cluster_labels)			# The silhouette_score gives the average value for all the samples. This gives a perspective into the density and separation of the formed clusters
	out.write('For n_clusters = %3d, The average silhouette_score is: %f\n' %(n_clusters,silhouette_avg))
	sample_silhouette_values = silhouette_samples(centered_matrix, cluster_labels)	# Compute the silhouette scores for each sample

	out1 = open('%02d.Cluster_labels.dat' %(n_clusters),'w')
	for i in range(len(cluster_labels)):
		out1.write(' %d \n' %(cluster_labels[i]))
	out1.close()

	# Initialize the figure to be made...
	fig, (ax1, ax2) = plt.subplots(1, 2)
	fig.set_size_inches(18, 7)

	# The 1st subplot is the silhouette plot
	y_lower = 10
	for i in range(n_clusters):
		ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
		ith_cluster_silhouette_values.sort()
		size_cluster_i = ith_cluster_silhouette_values.shape[0]
		y_upper = y_lower + size_cluster_i

		color = cm.spectral(float(i) / n_clusters)
		ax1.fill_betweenx(np.arange(y_lower, y_upper),0, ith_cluster_silhouette_values,facecolor=color, edgecolor=color, alpha=0.7)
		ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))		# Label the silhouette plots with their cluster numbers at the middle
		y_lower = y_upper + 10  					# Compute the new y_lower for next plot; 10 for the 0 samples

	ax1.axvline(x=silhouette_avg, color="red", linestyle="--")		# The vertical line for average silhoutte score of all the values
	ax1.set_yticks([])  							# Clear the yaxis labels / ticks
	ax1.set_title("The silhouette plot for the various clusters.")
	ax1.set_xlabel("The silhouette coefficient values")
	ax1.set_ylabel("Cluster label")
	ax1.set_xlim([-0.2,1.0])
	ax1.set_ylim([0, nSteps + (n_clusters + 1) * 10])			# The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.

	# 2nd Plot showing the actual clusters formed
	colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
	ax2.scatter(data1[:], data2[:], marker='.', s=30, lw=0, alpha=0.7,c=colors)
	
	ax2.set_title("The visualization of the clustered data.")
	ax2.set_xlabel("Feature space for the 1st feature")
	ax2.set_ylabel("Feature space for the 2nd feature")
	
	plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" %(n_clusters),fontsize=14, fontweight='bold')
	plt.savefig('%02d_clustering.png' %(n_clusters))
	plt.close()

out.close()

summary()
