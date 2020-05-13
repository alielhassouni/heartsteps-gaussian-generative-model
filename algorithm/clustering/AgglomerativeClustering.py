#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jul 27 14:39:89 2018
@author: emgrua
"""
from sklearn.cluster import AgglomerativeClustering

class AgglomerativeCluster:
    def cluster(self, k, distances, affinity, linkage):
        return AgglomerativeClustering(k, affinity=affinity, linkage=linkage).fit_predict(distances)