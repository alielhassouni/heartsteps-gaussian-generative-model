#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 21 14:58:89 2018
@author: alielhassouni
"""

import numpy as np
from algorithm.clustering.Distance import LB_Keogh
from pprint import pprint
from copy import deepcopy
import random

class KMedoids:

    # https://github.com/salspaugh/machine_learning/blob/master/clustering/kmedoids.py
    def cluster_complete(self, distances, k=3):
        m = distances.shape[0]  # number of points

        # Pick k random medoids.
        curr_medoids = np.array([-1] * k)
        while not len(np.unique(curr_medoids)) == k:
            curr_medoids = np.array([random.randint(0, m - 1) for _ in range(k)])
        old_medoids = np.array([-1] * k)  # Doesn't matter what we initialize these to.
        new_medoids = np.array([-1] * k)

        # Until the medoids stop updating, do the following:
        while not ((old_medoids == curr_medoids).all()):
            # Assign each point to cluster with closest medoid.
            clusters = self.assign_points_to_clusters(curr_medoids, distances)

            # Update cluster medoids to be lowest cost point.
            for curr_medoid in curr_medoids:
                cluster = np.where(clusters == curr_medoid)[0]
                new_medoids[curr_medoids == curr_medoid] = self.compute_new_medoid(cluster, distances)

            old_medoids[:] = curr_medoids[:]
            curr_medoids[:] = new_medoids[:]

        return clusters, curr_medoids

    def cluster(self, distances, k=3):
        return self.cluster_complete(distances, k)[0]

    def assign_points_to_clusters(self, medoids, distances):
        distances_to_medoids = distances[:, medoids]
        clusters = medoids[np.argmin(distances_to_medoids, axis=1)]
        clusters[medoids] = medoids
        return clusters

    def compute_new_medoid(self, cluster, distances):
        mask = np.ones(distances.shape)
        mask[np.ix_(cluster, cluster)] = 0.
        cluster_distances = np.ma.masked_array(data=distances, mask=mask, fill_value=10e9)
        costs = cluster_distances.sum(axis=1)
        return costs.argmin(axis=0, fill_value=10e9)
