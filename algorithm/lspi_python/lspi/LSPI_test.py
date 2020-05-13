#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Nov 28 15:40:19 2017
@author: Ali el Hasouni 
"""

from sample import Sample
from domains import ChainDomain
from policy import Policy
import numpy as np
from basis_functions import FakeBasis, OneDimensionalPolynomialBasis, ExactBasis
from solvers import *
from lspi import *


import numpy as np
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])
plt.ion()

#  Define dummy example of representation of one sample during learning task
state = [0,1,2]
action = 2
reward = 0.05
next_state = [0,2,0]
absorb = True

# Print generated sample
sample = Sample(state, action, reward, next_state, absorb)
print(sample.action)


num_states = 20
reward_location = 2
failure_probability = .3
domain = ChainDomain(num_states, reward_location, failure_probability)

sample = domain.apply_action(0)
print(sample)

max_iterations_solver = 10
policy = Policy(OneDimensionalPolynomialBasis(1, 2), weights=np.array([1., 1, 2, 2]))
print(policy)


# Start test
sampling_policy = Policy(FakeBasis(2), .9, 0.1)

samples = []
for i in range(10000):
    action = policy.select_action(domain.current_state())
    samples.append(domain.apply_action(action))





random_policy_cum_rewards = np.sum([sample.reward for sample in samples])

solver = LSTDQSolver()

print(random_policy_cum_rewards)

initial_policy = Policy(ExactBasis(3, 2),.9,0)
learned_policy = learn(samples, initial_policy, solver)

#domain.reset()
cumulative_reward = 0

print(learned_policy.weights)
for i in range(1000):
     action = learned_policy.select_action(domain.current_state())
     sample = domain.apply_action(action)
     cumulative_reward += sample.reward
     plt.scatter(i, cumulative_reward)
     plt.show()
     plt.pause(0.01)

print(cumulative_reward)


