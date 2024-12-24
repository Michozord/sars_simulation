# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:25:10 2024

@author: Michal Trojanowski
"""

from scipy.stats import bernoulli, nbinom, uniform


# Global parameters of the simulation
# TODO: move these variables to Simulation class or somewhere
global rho
rho = 0.9   # probability of being traced
global R_0, p
R_0 = 2.5   # parameters of negative binomial distribution
p = 0.7
global T
T = 12*7    # end of the simulation


class Person():
    def __init__(self, infection_time: float, is_traced: bool, is_subclinical: bool):

        
        # Determine timeline for this person:
        #
        #   TIMELINE:
        #   -- x ------------- x ------------------ x --------->
        #   infection   |   symptoms    |       isolation
        #               |               |               
        #         incubation      onset to isolation delay
        #
        self.infection_time = infection_time
        self.is_traced = is_traced
        self.is_subclinical = is_subclinical
        incubation_period = 5.8     # TODO: should be random, not hard-coded; probability distr. to be determined
        self.symptoms_time = self.infection_time + incubation_period
        if self.is_traced:
            self.isolation_time = self.symptoms_time    # traced cases are isolated with no delay
        elif self.is_subclinical:
            self.isolation_time = float('inf')      # subclinical cases are never isolated
        else:
            onset_to_isolation_delay = 3.43     # TODO: should be random, not hard-coded; probability distr. to be determined
            self.isolation_time = self.symptoms_time + onset_to_isolation_delay
        
        # Generate new cases:
        number_of_new_cases = nbinom.rvs(n=R_0, p=p)
        new_cases = [self.infection_time + uniform.rvs(0, 7) for _ in range(number_of_new_cases)]   # TODO: replace uniform distr by serial intervals; probability distr. to be determined
        new_cases = [case for case in new_cases if case <= T]   # ignore cases infected after T
        # TODO: add new_cases to queue
    
    def infect(self)
        
        
        
        