# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:25:10 2024

@author: Michal Trojanowski
"""

from __future__ import annotations

from scipy.stats import bernoulli, nbinom, uniform
from queue import Queue


class Person():
    def __init__(self, simulation: Simulation, infection_time: float, is_traced: bool = None):

        self.simulation = simulation
        # Determine timeline for this person:
        #
        #   TIMELINE:
        #   -- x ------------- x ------------------ x --------->
        #   infection   |   symptoms    |       isolation
        #               |               |               
        #         incubation      onset to isolation delay
        #
        self.infection_time = infection_time
        if is_traced is None:
            self.is_traced = bool(bernoulli.rvs(self.simulation.rho))
        else:
            self.is_traced = is_traced
        self.is_subclinical = bool(bernoulli.rvs(self.simulation.subclinical_prob))
        incubation_period = 5.8     # TODO: should be random, not hard-coded; probability distr. to be determined
        self.symptoms_time = self.infection_time + incubation_period
        if self.is_traced:
            self.isolation_time = self.symptoms_time    # traced cases are isolated with no delay
        elif self.is_subclinical:
            self.isolation_time = float('inf')      # subclinical cases are never isolated
        else:
            onset_to_isolation_delay = 3.43     # TODO: should be random, not hard-coded; probability distr. to be determined
            self.isolation_time = self.symptoms_time + onset_to_isolation_delay
        
    
    def infect(self):
        number_of_new_cases = nbinom.rvs(n=self.simulation.R_0, p=self.simulation.p)
        new_infections = [self.infection_time + uniform.rvs(0.1, 7) for _ in range(number_of_new_cases)]   # TODO: replace uniform distr by serial intervals; probability distr. to be determined
        new_infections = [i for i in new_infections if i <= self.simulation.T]   # ignore cases infected after T
        # print(f"{len(new_infections)} new infections")
        for infection in new_infections:
            person = Person(self.simulation, infection_time=infection)
            self.simulation.new_case(person)
    
        
class Simulation():
    def __init__(self, T: float, initial_cases: int, rho: float, R_0: float, p: float, subclinical_prob: float):
        self.T = T
        self.R_0 = R_0
        self.p = p          # TODO: parameter p should be somehow computed (how???) 
        self.subclinical_prob = subclinical_prob
        self.rho = rho
        self.cases = []     # empty list to store all infected cases as Person objects
        self.queue = Queue()    # queue manages infections
        # Generate initial cases
        for _ in range(initial_cases):
            person = Person(self, infection_time=0, is_traced=False)
            self.cases.append(person)
            self.queue.put(person)
    
    def simulate(self):
        while not self.queue.empty():
            person = self.queue.get()
            person.infect()
            
    def new_case(self, person: Person):
        self.queue.put(person)
        self.cases.append(person)
        
            
        
if __name__ == "__main__":
    sim = Simulation(T=20, initial_cases=5, rho=0.5, R_0=2.5, p=0.5, subclinical_prob=0.1)
    sim.simulate()
    