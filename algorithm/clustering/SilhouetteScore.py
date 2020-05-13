# Wrapping the silhouette score function in to the process of finding the optimal number of clusters
from sklearn.metrics import silhouette_score


def best_silhouette_score(minimum, maximum, distances, cluster_method, metric):
        """
        Find best silhouette score.
        :param minimum: smallest number of clusters to test silhoutte on
        :param maximum: maximum number of clusters to test silhoutte on
        :param distances: precalculated distances
        :param cluster_method: clustering method
        :param metric: metric.
        :return: cluster assignments and metrics.
        """
        best_k = 0
        best_score = -1000000000
        best_clusters = None
        print(distances)
        for k in range(minimum, maximum):

            clusters = cluster_method(k=k)
            silhouette_avg = silhouette_score(distances, clusters, metric=metric)
            print(clusters)
            print("__________________________________________________________________________")
            print("For n_clusters =", k, "The average silhouette_score is :", silhouette_avg)
            print("__________________________________________________________________________")
            if silhouette_avg > best_score:
                best_score = silhouette_avg
                best_clusters = clusters
                best_k = k
                print("__________________________________________________________________________")
                print("Best K =", best_k, "The bestp average silhouette_score is :", best_score)
                print("__________________________________________________________________________")
        return best_score, best_clusters, best_k
