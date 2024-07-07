from kmeans_lib import *

class KMeans:

    def __init__(self, X, K):
        self.K = K
        self.num_samples, self.num_features = X.shape
        np.random.seed(10)
        self.max_iteration = 100
        self.X = X
        self.centroids = np.zeros((self.K, self.num_features))
        self.cluster_list = [[] for _ in range(self.K)]

    def random_centroids(self):
        for k in range(self.K):
            self.centroids[k] = self.X[np.random.choice(range(self.num_samples))]
        
        return self.centroids
    
    def clustering(self):
        cluster_list = [[] for _ in range(self.K)]
        for idx, point in enumerate(self.X):
            cent_idx = np.argmin(np.sqrt(np.sum((point - self.centroids)**2, axis=1)))
            cluster_list[cent_idx].append(idx)
        # print(cluster_list)
        return cluster_list
    
    def new_centroids(self):
        centroids = np.zeros_like(self.centroids)

        for idx, cluster in enumerate(self.cluster_list):
            centroid = np.mean(self.X[cluster], axis=0)
            centroids[idx] = centroid

        return centroids
    
    def fit(self):
        self.centroids = self.random_centroids()

        pre_centroids = np.zeros_like(self.centroids)
        for it in range(self.max_iteration):
            self.cluster_list = self.clustering()

            self.centroids = self.new_centroids()

            diff = self.centroids - pre_centroids

            if not diff.any():
                print('Early Stopped')
                break

            pre_centroids = self.centroids
        
        clustered_list = np.zeros(self.num_samples)
        for cluster_idx, cluster in enumerate(self.cluster_list):
            for c in cluster:
                clustered_list[c] = cluster_idx
        
        return clustered_list


if __name__ == "__main__":
    num_clusters = 3
    X, _ = make_blobs(n_samples=10, n_features=2, centers=num_clusters)

    kmeans = KMeans(X, num_clusters)
    cluster_list = kmeans.fit()

            
