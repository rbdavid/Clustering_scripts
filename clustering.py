#!/Library/Frameworks/Python.framework/Versions/2.7/bin/python
# ----------------------------------------
# USAGE:


# ----------------------------------------
# PREAMBLE:

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

range_n_clusters = [2,3,4,5,6,7,8,9,10]
important_eigenvectors = int(sys.argv[1])
frames = int(sys.argv[2])

# ----------------------------------------
# SUBROUTINES:

def ffprint(string):
	print '%s' %(string)
	flush()

# ----------------------------------------
# MAIN PROGRAM:

data = np.zeros((frames,important_eigenvectors),dtype=np.float64)

for i in range(important_eigenvectors):
	temp_data = np.loadtxt('%02d.projection.dat' %(i))
	data[:,i] = temp_data

for n_clusters in range_n_clusters:
	# Initialize the clusterer with desired kwargs
	clusterer = KMeans(n_clusters=n_clusters,init='k-means++',n_init=10,max_iter=300,tol=0.0001,precompute_distances='auto',verbose=0,n_jobs=1)
	cluster_labels = clusterer.fit_predict(data)
	silhouette_avg = silhouette_score(data, cluster_labels)			# The silhouette_score gives the average value for all the samples. This gives a perspective into the density and separation of the formed clusters
	print 'For n_clusters =', n_clusters,'The average silhouette_score is :', silhouette_avg
	sample_silhouette_values = silhouette_samples(data, cluster_labels)	# Compute the silhouette scores for each sample
	
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
	ax1.set_xlim([-0.2,1])
	ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])			# The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters, to demarcate them clearly.

	# 2nd Plot showing the actual clusters formed
	colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
	ax2.scatter(data[:, 0], data[:, 1], marker='.', s=30, lw=0, alpha=0.7,c=colors)
	
	centers = clusterer.cluster_centers_					# Labeling the clusters
	ax2.scatter(centers[:, 0], centers[:, 1],marker='o', c="white", alpha=1, s=200) 	# Draw white circles at cluster centers
	for i, c in enumerate(centers):						# Add label to the circles
		ax2.scatter(c[0], c[1], marker='$%d$' %(i), alpha=1, s=50)
	
	ax2.set_title("The visualization of the clustered data.")
	ax2.set_xlabel("Feature space for the 1st feature")
	ax2.set_ylabel("Feature space for the 2nd feature")
	
	plt.suptitle("Silhouette analysis for KMeans clustering on sample data with n_clusters = %d" %(n_clusters),fontsize=14, fontweight='bold')
	plt.savefig('clustering_%s.png' %(n_clusters))

