#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jan 21 09:09:09 2018
@author: alielhassouni
"""

import numpy as np
import math
from algorithm.clustering.Distance import LB_Keogh
from algorithm.clustering.KMeans import KMeans


class Cluster:

    def __init__(self, k_range, agent_profiles_params, verbose = False):
        self.verbose = verbose
        self.k_range = k_range
        self.agent_profiles_params = agent_profiles_params
        self.dtw_days_matching_of_profiles = \
            self.get_sorted_average_amount_activity_per_day_per_profile(agent_profiles_params)

    def get_sorted_average_amount_activity_per_day_per_profile(self, agent_profiles_params):
        result = dict()

        for profile in agent_profiles_params["activity_profiles"].keys():
            a = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for activity in agent_profiles_params["activity_profiles"][profile].keys():
                b = agent_profiles_params["activity_profiles"] \
                    [profile][activity]["probability_performing_activity_per_weekday"]
                a = np.add((a), (b))
            a = np.argsort(a)
            result.update({profile: a[::-1]})

        return result

    def cluster(self, data, profiles):
        kmeans = KMeans(k=9, tolerance=0.001, max_iterations=100)
        kmeans.kmeans(data, profiles, self.agent_profiles_params, self.dtw_days_matching_of_profiles)
        return kmeans