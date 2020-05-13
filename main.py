#!/usr/bin/env python3
# Created by Ali el Hassouni

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from algorithm.lspi_python.lspi.sample import Sample
from algorithm.lspi_python.lspi.policy import Policy
from algorithm.lspi_python.lspi.basis_functions import OneDimensionalPolynomialBasis, ExactBasis
from algorithm.lspi_python.lspi.solvers import *
from algorithm.lspi_python.lspi.lspi import *
from algorithm.QLearning.qlearn import *
from copy import deepcopy
import functools
from algorithm.clustering.Distance import LB_Keogh
from algorithm.clustering.KMedoids import KMedoids
from algorithm.clustering.SilhouetteScore import best_silhouette_score
from sklearn.preprocessing import normalize
from scipy.spatial import distance
import pandas as pd

"""
Implementation of https://arxiv.org/pdf/1708.04001.pdf.
"""


class GroupDrivenRL:
    actions: List[int] = [0, 1]  # 0: no intervention, 1: positive intervention
    p: int = 3
    q: int = 4

    M: int = 5  # number of groups
    N_m: int = 20  # number of people per group
    N = M * N_m  # Total number of people

    zeta_a: float = 0.01
    zeta_c: float = 0.01

    sigma_b: float = 0.01
    sigma_s: float = 1
    sigma_r: float = 1

    Beta = [1] * 14

    T = 100
    T_simulate = 4000

    n_bins = 15

    algorithm = "Q"

    def generate_user_noise(self):
        """
        Generate noise for state from normal distribution
        :param sigma_b: stdev
        :return: noise
        """
        return np.random.normal(0, self.sigma_b, 1)

    def generate_state_noise(self):
        """
        Generate noise for state from normal distribution
        :param sigma_s: stdev
        :return: noise
        """
        return np.random.normal(0, self.sigma_s, 1)

    def generate_reward_noise(self):
        """
        Generate noise for state from normal distribution
        :param sigma_r: stdev
        :return: noise
        """
        return np.random.normal(0, self.sigma_r, 1)

    def sample_initial_state(self):
        """
        Sample the initial state S0
        :param p:
        :param sigma_s: pxp covariance matrix with predefined elements.
        :return:
        """
        return self.generate_state_noise()

    def select_action(self):
        """
        Select next action.
        :param s: The state.
        :return: The action.
        """
        p = random.random()
        if p >= 0.5:
            return 1
        else:
            return 0


@dataclass
class BetaBasic:

    beta_basic_1 = [0.40, 0.25, 0.35, 0.65, 0.10, 0.50, 0.22, 2.00, 0.15, 0.20, 0.32, 0.10, 0.45, 800]
    beta_basic_2 = [0.45, 0.35, 0.40, 0.70, 0.15, 0.55, 0.30, 2.20, 0.25, 0.25, 0.40, 0.12, 0.55, 700]
    beta_basic_3 = [0.35, 0.30, 0.30, 0.60, 0.05, 0.65, 0.28, 2.60, 0.35, 0.45, 0.45, 0.15, 0.50, 650]
    beta_basic_4 = [0.55, 0.40, 0.25, 0.55, 0.08, 0.70, 0.26, 3.10, 0.25, 0.35, 0.30, 0.17, 0.60, 500]
    beta_basic_5 = [0.20, 0.50, 0.20, 0.62, 0.06, 0.52, 0.27, 3.00, 0.15, 0.15, 0.50, 0.16, 0.70, 450]

    def get_beta_basic(self, m):
        """
        Return beta basic for group number.
        :param m: group number.
        :return: List.
        """
        if m == 1:
            return self.beta_basic_1
        elif m == 2:
            return self.beta_basic_2
        elif m == 3:
            return self.beta_basic_3
        elif m == 4:
            return self.beta_basic_4
        elif m == 5:
            return self.beta_basic_5


class LSPI_class():
    def __init__(self, n_user):
        self.LSPI_samples = []
        self.LSPI_samples_user = [[]]*n_user
        self.bins = None
        self.n_bins = 15

    def append_sample(self, sample):
        """
        Append new states, actions, reward and next_states.
        :param sample: new sample [S_1_prev, S_2_prev, S_3_prev, action_prev, reward_prev, S_1, S_2, S_3]
        """
        states = sample[0:3]
        actions = sample[3]
        rewards = sample[4]
        next_states = sample[5:9]

        sample = Sample(
            self.get_feature_approximation_state([self.n_bins]*3, np.array(states)-1),
            actions,
            np.float(rewards),
            self.get_feature_approximation_state([self.n_bins]*3, np.array(next_states)-1),
            absorb=False)

        self.LSPI_samples.append(sample)

    def append_sample_user(self, sample, user):
        """
        Append new states, actions, reward and next_states per user.
        :param user: user id.
        :param sample: new sample [S_1_prev, S_2_prev, S_3_prev, action_prev, reward_prev, S_1, S_2, S_3]
        """
        states = sample[0:3]
        actions = sample[3]
        rewards = sample[4]
        next_states = sample[5:9]

        sample = Sample(
                self.get_feature_approximation_state([self.n_bins]*3, np.array(states) - 1),
                actions,
                np.float(rewards),
                self.get_feature_approximation_state([self.n_bins]*3, np.array(next_states) - 1),
                absorb=False)

        self.LSPI_samples_user[user].append(sample)

    def get_feature_approximation_state(self, n_features, state):
        """
        Return basis function features for LSPI.
        :param n_features: the number of features in the state.
        :param state: The state values.
        :return:
        """
        result = np.zeros(sum(n_features), dtype=int)
        for i in range(0, len(n_features)):
            first_position = int(np.sum(n_features[0:i]))
            result[first_position + int(state[i])] = 1
        return result

    def learn(self):
        """
        Learn policy using all the available samples.
        :return: policy object containing the learned weights.
        """
        basis = ExactBasis(np.array([2]*self.n_bins*3), 2)
        policy = Policy(basis=basis,
                        discount=0.95,
                        explore=0.1,
                        tie_breaking_strategy=Policy.TieBreakingStrategy.FirstWins)
        solver = LSTDQSolver()
        p = learn(self.LSPI_samples,
                  policy,
                  solver,
                  max_iterations=1000,
                  epsilon=0.00001)
        return p

    def learn_user(self, users):
        """
        Learn policy using all the available samples.
        :param users: the users that will be inlcluded in the learning step.
        :return: policy object containing the learned weights.
        """
        basis = ExactBasis(np.array([2]*self.n_bins*3), 2)
        policy = Policy(basis=basis,
                        discount=0.95,
                        explore=0.1,
                        tie_breaking_strategy=Policy.TieBreakingStrategy.FirstWins)
        solver = LSTDQSolver()

        s = np.array(self.LSPI_samples_user)
        samples = s[users]

        p = learn(samples.flatten(),
                  policy,
                  solver,
                  max_iterations=1000,
                  epsilon=0.00001)
        return p


class Q_class():

    def __init__(self, n_user):
        self.Q_samples = []
        self.Q_samples_user = [[]]*n_user
        self.bins = None
        self.n_bins = 15

    def append_sample(self, sample):
        """
        Append new states, actions, reward and next_states.
        :param sample: new sample [S_1_prev, S_2_prev, S_3_prev, action_prev, reward_prev, S_1, S_2, S_3]
        """
        self.Q_samples.append(sample)

    def append_sample_user(self, sample, user):
        """
        Append new states, actions, reward and next_states per user.
        :param user: user id.
        :param sample: new sample [S_1_prev, S_2_prev, S_3_prev, action_prev, reward_prev, S_1, S_2, S_3]
        """
        self.Q_samples_user[user].append(sample)

    def learn(self):
        """
        Learn policy using all the available samples.
        :return: policy object containing the learned weights.
        """
        Q = QLearn([0, 1],
                   epsilon=0.2,
                   alpha=0.2,
                   gamma=0.99)
        for sample in self.Q_samples:
            Q.learn(hash(tuple(sample[0])), int(sample[1]), float(sample[2]), hash(tuple(sample[3])))
        return Q

    def learn_user(self, users):
        """
        Learn policy using all the available samples.
        :param users: the users that will be inlcluded in the learning step.
        :return: policy object containing the learned weights.
        """
        Q = QLearn([0, 1],
                   epsilon=0.05,  # AMin, increasing the probability of exploring
                   alpha=0.2,
                   gamma=0.99)
        s = np.array(self.Q_samples_user)
        samples = s[users]
        for sample in samples:
            Q.learn(hash(tuple(sample[0][0])), int(sample[0][1]), float(sample[0][2]), hash(tuple(sample[0][3])))
        return Q


def pooled():
    """
    Learn a policy across all users
    """
    GDRL = GroupDrivenRL()
    BB = BetaBasic()
    LSPI = LSPI_class(GDRL.M * GDRL.N_m)
    Q = Q_class(GDRL.M * GDRL.N_m)

    s_1 = []
    s_2 = []
    s_3 = []

    bins = np.linspace(-6, 6, GDRL.n_bins)
    LSPI.bins = bins

    states = [None] * GDRL.M * GDRL.N_m

    for t in range(GDRL.T):
        for m in range(GDRL.M):
            for i in range(GDRL.N_m):
                user_noise = GDRL.generate_user_noise()
                beta_i = BB.get_beta_basic(m + 1) + user_noise

                if t == 0:
                    action = GDRL.select_action()
                    S_1 = beta_i[0]*GDRL.sample_initial_state() + GDRL.generate_state_noise()
                    S_2 = beta_i[1]*GDRL.sample_initial_state() + beta_i[2]*action + GDRL.generate_state_noise()
                    S_3 = beta_i[3]*GDRL.sample_initial_state() + beta_i[4]*GDRL.sample_initial_state()*action + \
                          beta_i[5]*action + GDRL.generate_state_noise()[0]

                    s_1.append(S_1)
                    s_2.append(S_2)
                    s_3.append(S_3)

                    part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]
                    part_b = (beta_i[7] + action * part_a + beta_i[11]*S_1[0] - beta_i[12]*S_3[0] )
                    part_c = GDRL.generate_reward_noise()[0]

                    R = beta_i[13] * part_b + part_c
                    SAR = [S_1[0], S_2[0], S_3[0], action, R]
                    states[m*GDRL.N_m + i] = SAR

                else:
                    action = GDRL.select_action()
                    S_1_prev = states[m * GDRL.N_m + i][0]
                    S_2_prev = states[m * GDRL.N_m + i][1]
                    S_3_prev = states[m * GDRL.N_m + i][2]
                    action_prev = states[m * GDRL.N_m + i][3]
                    reward_prev = states[m * GDRL.N_m + i][4]

                    S_1 = beta_i[0] * S_1_prev + GDRL.generate_state_noise()
                    S_2 = beta_i[1] * S_2_prev + beta_i[2] * action_prev + GDRL.generate_state_noise()
                    S_3 = beta_i[3] * S_3_prev + beta_i[4] * S_3_prev * action_prev + \
                          beta_i[5] * action_prev + GDRL.generate_state_noise()[0]

                    part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]
                    part_b = beta_i[7] + action * part_a + beta_i[11] * S_1[0] - beta_i[12] * S_3
                    part_c = GDRL.generate_reward_noise()[0]
                    R = beta_i[13] * part_b + part_c
                    SAR = [S_1[0], S_2[0], S_3, action, R]
                    states[m * GDRL.N_m + i] = SAR

                    LSPI.append_sample([
                        np.digitize(S_1_prev, bins),
                        np.digitize(S_2_prev, bins),
                        np.digitize(S_3_prev, bins),
                        action_prev,
                        reward_prev,
                        np.digitize(S_1[0], bins),
                        np.digitize(S_2[0], bins),
                        np.digitize(S_3, bins)])

                    Q.append_sample([
                        [np.digitize(S_1_prev, bins).tolist(),
                         np.digitize(S_2_prev, bins).tolist(),
                         np.digitize(S_3_prev, bins).tolist()],
                        action_prev,
                        reward_prev,
                        [np.digitize(S_1[0], bins).tolist(),
                         np.digitize(S_2[0], bins).tolist(),
                         np.digitize(S_3, bins).tolist()]])

    result = LSPI.learn()
    Q_policy = Q.learn()

    state = [np.digitize(S_1_prev, bins),
             np.digitize(S_2_prev, bins),
             np.digitize(S_3_prev, bins),
             action_prev,
             reward_prev,
             np.digitize(S_1[0], bins),
             np.digitize(S_2[0], bins),
             np.digitize(S_3, bins)]

    reward = []
    for t in range(GDRL.T_simulate):
        for m in range(GDRL.M):
            for i in range(GDRL.N_m):
                user_noise = GDRL.generate_user_noise()
                beta_i = BB.get_beta_basic(m + 1) + user_noise

                if GDRL.algorithm == "LSPI":
                    state = [np.digitize(S_1_prev, bins),
                             np.digitize(S_2_prev, bins),
                             np.digitize(S_3_prev, bins),
                             action_prev,
                             reward_prev,
                             np.digitize(S_1[0], bins),
                             np.digitize(S_2[0], bins),
                             np.digitize(S_3, bins)]
                    action = result.select_action(np.array(
                        LSPI.get_feature_approximation_state(
                            [GDRL.n_bins]*3, np.array(state[5:9]) - 1)))

                elif GDRL.algorithm == "Q":
                    state_Q = [[np.digitize(S_1_prev, bins).tolist(),
                                np.digitize(S_2_prev, bins).tolist(),
                                np.digitize(S_3_prev, bins).tolist()],
                               action_prev,
                               reward_prev,
                               [np.digitize(S_1[0], bins).tolist(),
                                np.digitize(S_2[0], bins).tolist(),
                                np.digitize(S_3, bins).tolist()]]
                    action = Q_policy.chooseAction(hash(tuple(np.array(state_Q[5:9]))))

                S_1_prev = states[m * GDRL.N_m + i][0]
                S_2_prev = states[m * GDRL.N_m + i][1]
                S_3_prev = states[m * GDRL.N_m + i][2]
                action_prev = states[m * GDRL.N_m + i][3]
                reward_prev = states[m * GDRL.N_m + i][4]

                S_1 = beta_i[0] * S_1_prev + GDRL.generate_state_noise()
                S_2 = beta_i[1] * S_2_prev + beta_i[2] * action_prev + GDRL.generate_state_noise()
                S_3 = beta_i[3] * S_3_prev + beta_i[4] * S_3_prev * action_prev + \
                      beta_i[5] * action_prev + GDRL.generate_state_noise()[0]

                part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]
                part_b = beta_i[7] + action * part_a + beta_i[11] * S_1[0] - beta_i[12] * S_3
                part_c = GDRL.generate_reward_noise()[0]
                R = beta_i[13] * part_b + part_c
                reward.append(R)
                SAR = [S_1[0], S_2[0], S_3, action, R]
                states[m * GDRL.N_m + i] = SAR

    print("Algorithm: " +  str(GDRL.algorithm))
    print("T: " + str(GDRL.T))
    print("T evaluation: " + str(GDRL.T_simulate))
    print("Mean Average Reward: " + str(sum(reward)/len(reward)))


def separate():
    """
    Learn one policy per user.
    """
    GDRL = GroupDrivenRL()
    BB = BetaBasic()
    LSPI = LSPI_class(GDRL.M * GDRL.N_m)
    Q = Q_class(GDRL.M * GDRL.N_m)

    s_1 = []
    s_2 = []
    s_3 = []

    bins = np.linspace(-6, 6, GDRL.n_bins)
    LSPI.bins = bins

    reward_per_user = []
    for m in range(GDRL.M):
        for i in range(GDRL.N_m):
            states = [None]
            LSPI = LSPI_class(GDRL.M * GDRL.N_m)
            Q = Q_class(GDRL.M * GDRL.N_m)

            for t in range(GDRL.T):
                user_noise = GDRL.generate_user_noise()
                beta_i = BB.get_beta_basic(m + 1) + user_noise

                if t == 0:
                    action = GDRL.select_action()
                    S_1 = beta_i[0]*GDRL.sample_initial_state() + GDRL.generate_state_noise()
                    S_2 = beta_i[1]*GDRL.sample_initial_state() + beta_i[2]*action + GDRL.generate_state_noise()
                    S_3 = beta_i[3]*GDRL.sample_initial_state() + beta_i[4]*GDRL.sample_initial_state()*action + \
                          beta_i[5]*action + GDRL.generate_state_noise()[0]

                    s_1.append(S_1)
                    s_2.append(S_2)
                    s_3.append(S_3)

                    part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]
                    part_b = (beta_i[7] + action * part_a + beta_i[11]*S_1[0] - beta_i[12]*S_3[0])
                    part_c = GDRL.generate_reward_noise()[0]

                    R = beta_i[13] * part_b + part_c
                    SAR = [S_1[0], S_2[0], S_3[0], action, R]
                    states[0] = SAR

                else:
                    action = GDRL.select_action()
                    S_1_prev = states[0][0]
                    S_2_prev = states[0][1]
                    S_3_prev = states[0][2]
                    action_prev = states[0][3]
                    reward_prev = states[0][4]

                    S_1 = beta_i[0] * S_1_prev + GDRL.generate_state_noise()
                    S_2 = beta_i[1] * S_2_prev + beta_i[2] * action_prev + GDRL.generate_state_noise()
                    S_3 = beta_i[3] * S_3_prev + beta_i[4] * S_3_prev * action_prev + \
                          beta_i[5] * action_prev + GDRL.generate_state_noise()[0]

                    part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]

                    part_b = beta_i[7] + action * part_a + beta_i[11] * S_1[0] - beta_i[12] * S_3
                    part_c = GDRL.generate_reward_noise()[0]
                    R = beta_i[13] * part_b + part_c
                    SAR = [S_1[0], S_2[0], S_3, action, R]

                    states[0] = SAR

                    LSPI.append_sample(
                        [np.digitize(S_1_prev, bins),
                         np.digitize(S_2_prev, bins),
                         np.digitize(S_3_prev, bins),
                         action_prev,
                         reward_prev,
                         np.digitize(S_1[0], bins),
                         np.digitize(S_2[0], bins),
                         np.digitize(S_3, bins)])

                    Q.append_sample(
                        [[np.digitize(S_1_prev, bins).tolist(),
                          np.digitize(S_2_prev, bins).tolist(),
                          np.digitize(S_3_prev, bins).tolist()],
                         action_prev,
                         reward_prev,
                         [np.digitize(S_1[0], bins).tolist(),
                          np.digitize(S_2[0], bins).tolist(),
                          np.digitize(S_3, bins).tolist()]])

            result = LSPI.learn()
            Q_policy = Q.learn()

            state = [np.digitize(S_1_prev, bins),
                     np.digitize(S_2_prev, bins),
                     np.digitize(S_3_prev, bins),
                     action_prev,
                     reward_prev,
                     np.digitize(S_1[0], bins),
                     np.digitize(S_2[0], bins),
                     np.digitize(S_3, bins)]

            state = LSPI.get_feature_approximation_state([GDRL.n_bins]*3, np.array(state[5:9]) - 1)

            reward = []
            for t in range(GDRL.T_simulate):
                        user_noise = GDRL.generate_user_noise()
                        beta_i = BB.get_beta_basic(m + 1) + user_noise

                        if GDRL.algorithm == "LSPI":
                            state = [np.digitize(S_1_prev, bins),
                                     np.digitize(S_2_prev, bins),
                                     np.digitize(S_3_prev, bins),
                                     action_prev,
                                     reward_prev,
                                     np.digitize(S_1[0], bins),
                                     np.digitize(S_2[0], bins),
                                     np.digitize(S_3, bins)]
                            action = result.select_action(
                                np.array(
                                    LSPI.get_feature_approximation_state(
                                        [GDRL.n_bins]*3, np.array(state[5:9]) - 1)))

                        elif GDRL.algorithm == "Q":
                            state_Q = [[np.digitize(S_1_prev, bins).tolist(), np.digitize(S_2_prev, bins).tolist(),
                                        np.digitize(S_3_prev, bins).tolist()],
                                       action_prev, reward_prev,
                                       [np.digitize(S_1[0], bins).tolist(), np.digitize(S_2[0], bins).tolist(),
                                        np.digitize(S_3, bins).tolist()]]
                            action = Q_policy.chooseAction(hash(tuple(np.array(state_Q[5:9]))))

                        S_1_prev = states[0][0]
                        S_2_prev = states[0][1]
                        S_3_prev = states[0][2]
                        action_prev = states[0][3]
                        reward_prev = states[0][4]

                        S_1 = beta_i[0] * S_1_prev + GDRL.generate_state_noise()
                        S_2 = beta_i[1] * S_2_prev + beta_i[2] * action_prev + GDRL.generate_state_noise()
                        S_3 = beta_i[3] * S_3_prev + beta_i[4] * S_3_prev * action_prev + \
                              beta_i[5] * action_prev + GDRL.generate_state_noise()[0]

                        part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]
                        part_b = beta_i[7] + action * part_a + beta_i[11] * S_1[0] - beta_i[12] * S_3
                        part_c = GDRL.generate_reward_noise()[0]
                        R = beta_i[13] * part_b + part_c
                        reward.append(R)
                        SAR = [S_1[0], S_2[0], S_3, action, R]
                        states[0] = SAR

            print("Algorithm: " +  str(GDRL.algorithm))
            print("T: " + str(GDRL.T))
            print("T evaluation: " + str(GDRL.T_simulate))
            print("Mean Average Reward: " + str(sum(reward)/len(reward)))

            reward_per_user.append(sum(reward)/len(reward))
    print("Mean Average total Reward: " + str(sum(reward_per_user) / len(reward_per_user)))


def grouped():
    """
    Learn one policy per group of users.
    """
    GDRL = GroupDrivenRL()
    BB = BetaBasic()
    LSPI = LSPI_class(GDRL.M * GDRL.N_m)
    Q = Q_class(GDRL.M * GDRL.N_m)

    s_1 = []
    s_2 = []
    s_3 = []

    bins = np.linspace(-6, 6, GDRL.n_bins)
    LSPI.bins = bins

    states = [None] * GDRL.M * GDRL.N_m

    clustering_data = [[]] * GDRL.M * GDRL.N_m
    clustering_data_s1 = [[]] * GDRL.M * GDRL.N_m
    clustering_data_s2 = [[]] * GDRL.M * GDRL.N_m
    clustering_data_s3 = [[]] * GDRL.M * GDRL.N_m
    clustering_data_r = [[]] * GDRL.M * GDRL.N_m

    for t in range(GDRL.T):
        for m in range(GDRL.M):
            for i in range(GDRL.N_m):

                user_noise = GDRL.generate_user_noise()
                beta_i = BB.get_beta_basic(m + 1) + user_noise

                if t == 0:
                    action = GDRL.select_action()
                    S_1 = beta_i[0]*GDRL.sample_initial_state() + GDRL.generate_state_noise()
                    S_2 = beta_i[1]*GDRL.sample_initial_state() + beta_i[2]*action + GDRL.generate_state_noise()
                    S_3 = beta_i[3]*GDRL.sample_initial_state() + beta_i[4]*GDRL.sample_initial_state()*action + \
                          beta_i[5]*action + GDRL.generate_state_noise()[0]

                    s_1.append(S_1)
                    s_2.append(S_2)
                    s_3.append(S_3)

                    part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]
                    part_b = (beta_i[7] + action * part_a + beta_i[11]*S_1[0] - beta_i[12]*S_3[0] )
                    part_c = GDRL.generate_reward_noise()[0]

                    R = beta_i[13] * part_b + part_c
                    SAR = [S_1[0], S_2[0], S_3[0], action, R]

                    states[m * GDRL.N_m + i] = SAR

                    clustering_data[m * GDRL.N_m + i].append(S_1[0])
                    clustering_data[m * GDRL.N_m + i].append(S_2[0])
                    clustering_data[m * GDRL.N_m + i].append(S_3[0])
                    clustering_data[m * GDRL.N_m + i].append(R)

                    clustering_data_s1[m * GDRL.N_m + i].append(S_1[0])
                    clustering_data_s2[m * GDRL.N_m + i].append(S_2[0])
                    clustering_data_s3[m * GDRL.N_m + i].append(S_3[0])
                    clustering_data_r[m * GDRL.N_m + i].append(R)

                else:
                    action = GDRL.select_action()
                    S_1_prev = states[m * GDRL.N_m + i][0]
                    S_2_prev = states[m * GDRL.N_m + i][1]
                    S_3_prev = states[m * GDRL.N_m + i][2]
                    action_prev = states[m * GDRL.N_m + i][3]
                    reward_prev = states[m * GDRL.N_m + i][4]

                    S_1 = beta_i[0] * S_1_prev + GDRL.generate_state_noise()
                    S_2 = beta_i[1] * S_2_prev + beta_i[2] * action_prev + GDRL.generate_state_noise()
                    S_3 = beta_i[3] * S_3_prev + beta_i[4] * S_3_prev * action_prev + \
                          beta_i[5] * action_prev + GDRL.generate_state_noise()[0]

                    part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]

                    part_b = beta_i[7] + action * part_a + beta_i[11] * S_1[0] - beta_i[12] * S_3
                    part_c = GDRL.generate_reward_noise()[0]
                    R = beta_i[13] * part_b + part_c
                    SAR = [S_1[0], S_2[0], S_3, action, R]

                    states[m * GDRL.N_m + i] = SAR
                    clustering_data[m*GDRL.N_m + i].append(np.digitize(S_1[0], bins))
                    clustering_data[m*GDRL.N_m + i].append(np.digitize(S_2[0], bins))
                    clustering_data[m*GDRL.N_m + i].append(np.digitize(S_3, bins))
                    clustering_data[m*GDRL.N_m + i].append(R)

                    clustering_data_s1[m * GDRL.N_m + i].append(S_1[0])
                    clustering_data_s2[m * GDRL.N_m + i].append(S_2[0])
                    clustering_data_s3[m * GDRL.N_m + i].append(S_3)
                    clustering_data_r[m * GDRL.N_m + i].append(R)

                    LSPI.append_sample_user(
                        [np.digitize(S_1_prev, bins),
                         np.digitize(S_2_prev, bins),
                         np.digitize(S_3_prev, bins),
                         action_prev,
                         reward_prev,
                         np.digitize(S_1[0], bins),
                         np.digitize(S_2[0], bins),
                         np.digitize(S_3, bins)],
                        m*GDRL.N_m + i)

                    Q.append_sample_user(
                        [[np.digitize(S_1_prev, bins).tolist(),
                          np.digitize(S_2_prev, bins).tolist(),
                          np.digitize(S_3_prev, bins).tolist()],
                         action_prev, reward_prev,
                         [np.digitize(S_1[0], bins).tolist(),
                          np.digitize(S_2[0], bins).tolist(),
                         np.digitize(S_3, bins).tolist()]],
                        m*GDRL.N_m + i)

    clustering_data_s1_np = pd.DataFrame.from_records(clustering_data_s1)
    clustering_data_s2_np = pd.DataFrame.from_records(clustering_data_s2)
    clustering_data_s3_np = pd.DataFrame.from_records(clustering_data_s3)
    clustering_data_r_np = pd.DataFrame.from_records(clustering_data_r)

    distances_s1 = pre_calculate_distances(clustering_data_s1_np, norm=False)
    distances_s2 = pre_calculate_distances(clustering_data_s2_np, norm=False)
    distances_s3 = pre_calculate_distances(clustering_data_s3_np, norm=False)
    distances_r = pre_calculate_distances(clustering_data_r_np, norm=False)

    final_distances = (distances_s1 + distances_s2 + distances_s3 + distances_r)/4

    print("K means")
    cluster = functools.partial(KMedoids().cluster, distances=final_distances)
    minimum = 3
    maximum = 7
    best_score, best_clusters, best_k = best_silhouette_score(minimum, maximum, final_distances, cluster,
                                                              metric="precomputed")

    print("final best score: " + str(best_score))
    print("final best clusters: " + str(best_clusters))
    print("final best k: " + str(best_k))

    beta_number = [None]*(GDRL.M*GDRL.N_m)
    y = 0
    for b in range(GDRL.M):
        for q in range(GDRL.N_m):
            beta_number[y] = b + 1
            y = y + 1

    rewards = []
    for g in unique(best_clusters):
        indices = [i for i, x in enumerate(best_clusters.tolist()) if x == g]

        if GDRL.algorithm == "LSPI":
            result = LSPI.learn_user(indices)
        else:
            Q_policy = Q.learn_user(indices)

        reward = []
        for t in range(GDRL.T_simulate):
                for k in indices:
                    user_noise = GDRL.generate_user_noise()
                    beta_i = BB.get_beta_basic(beta_number[k]) + user_noise

                    S_1_prev = states[k][0]
                    S_2_prev = states[k][1]
                    S_3_prev = states[k][2]
                    action_prev = states[k][3]
                    reward_prev = states[k][4]

                    if GDRL.algorithm == "LSPI":
                        state = [np.digitize(S_1_prev, bins), np.digitize(S_2_prev, bins), np.digitize(S_3_prev, bins),
                                 action_prev, reward_prev, np.digitize(S_1[0], bins), np.digitize(S_2[0], bins),
                                 np.digitize(S_3, bins)]
                        action = result.select_action(
                            np.array(LSPI.get_feature_approximation_state([GDRL.n_bins]*3, np.array(state[5:9]) - 1)))
                    elif GDRL.algorithm == "Q":
                        state_Q = [[np.digitize(S_1_prev, bins).tolist(), np.digitize(S_2_prev, bins).tolist(),
                                    np.digitize(S_3_prev, bins).tolist()],
                                   action_prev, reward_prev,
                                   [np.digitize(S_1[0], bins).tolist(), np.digitize(S_2[0], bins).tolist(),
                                    np.digitize(S_3, bins).tolist()]]
                        action = Q_policy.chooseAction(hash(tuple(np.array(state_Q[5:9]))))

                    S_1_prev = states[k][0]
                    S_2_prev = states[k][1]
                    S_3_prev = states[k][2]
                    action_prev = states[k][3]
                    reward_prev = states[k][4]

                    S_1 = beta_i[0] * S_1_prev + GDRL.generate_state_noise()
                    S_2 = beta_i[1] * S_2_prev + beta_i[2] * action_prev + GDRL.generate_state_noise()
                    S_3 = beta_i[3] * S_3_prev + beta_i[4] * S_3_prev * action_prev + \
                          beta_i[5] * action_prev + GDRL.generate_state_noise()[0]

                    part_a = beta_i[8] + beta_i[9] * S_1[0] + beta_i[10] * S_2[0]
                    part_b = beta_i[7] + action * part_a + beta_i[11] * S_1[0] - beta_i[12] * S_3
                    part_c = GDRL.generate_reward_noise()[0]
                    R = beta_i[13] * part_b + part_c
                    reward.append(R)
                    rewards.append(R)

                    SAR = [S_1[0], S_2[0], S_3, action, R]
                    states[k] = SAR

        print("Algorithm: " +  str(GDRL.algorithm))
        print("T: " + str(GDRL.T))
        print("T evaluation: " + str(GDRL.T_simulate))
        print("Mean Average Reward: " + str(sum(reward)/len(reward)))

    print("Final Algorithm: " + str(GDRL.algorithm))
    print("Final T: " + str(GDRL.T))
    print("Final T evaluation: " + str(GDRL.T_simulate))
    print("Final Mean Average Reward: " + str(sum(rewards) / len(rewards)))


def unique(A):
    """
    :return: return unique values.
    """
    unique_values = []

    for i in A:
        if i not in unique_values:
            unique_values.append(i)
    return unique_values


def pre_calculate_distances(traces, norm=False):
        """
        Pre-calculate the distance matrix for clustering.
        """
        dist = "Euclidean"
        if dist == 'dtw':
            LBKeogh = LB_Keogh()
            distances = np.zeros(shape=(len(traces), len(traces)))
            for i in range(0, len(traces)):
                for j in range(0, len(traces)):
                    if i == j:
                        distances[i, j] = 0
                    elif i < j:
                        #dist = LBKeogh.dynamic_time_warping_new(traces[i].values, traces[j].values)
                        dist = LBKeogh.LB_Keogh(s1=traces[i], s2=traces[j], r=1)
                        distances[i, j] = dist
                        distances[j, i] = dist
            if norm:
                distances = normalize(distances, axis=1, norm='l1')

            return distances

        elif dist == 'Euclidean':
            distances = np.zeros(shape=(len(traces), len(traces)))

            for i in range(0, len(traces)):
                for j in range(0, len(traces)):
                    if i == j:
                        distances[i, j] = 0
                    elif i < j:
                        dist = distance.euclidean(traces[i], traces[j])
                        distances[i, j] = dist
                        distances[j, i] = dist
            if norm:
                distances = normalize(distances, axis=1, norm='l1')
            return distances


def main():
    pooled()
    separate()
    grouped()


if __name__ == "__main__":
    main()

