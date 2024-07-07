import kmeans_lib
from kmeans import *
from kmeans_plot import *

num_clusters = 3
X, _ = make_blobs(n_samples=1000, n_features=2, centers=num_clusters)

kmeans = KMeans(X, num_clusters)
cluster_list = kmeans.fit()

plot = KMeansPlot(X, cluster_list)
plot.plot()
