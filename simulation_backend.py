# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:25:10 2024

@author: Michal Trojanowski
"""

from __future__ import annotations

from scipy.stats import bernoulli, nbinom, uniform, weibull_min, skewnorm
from scipy.optimize import minimize
from queue import Queue


class Person:
    def __init__(
        self,
        scenario: Scenario,
        infection_time: float,
        is_traced: bool = None,
    ):
        self.scenario = scenario
        # Determine timeline for this person:
        #
        #   TIMELINE:
        #   -- x ------------- x ------------------ x --------->
        #   infection   |   symptoms    |       isolation
        #               |               |
        #         incubation      onset to isolation delay
        #
        if is_traced is None:
            self.is_traced = bool(bernoulli.rvs(self.scenario.rho))
        else:
            self.is_traced = is_traced
        self.is_subclinical = bool(bernoulli.rvs(self.scenario.subclinical_prob))
        self.infection_time = infection_time 
        self.incubation_period = weibull_min.rvs(self.scenario.incubation_p_shape, scale=self.scenario.incubation_p_scale)
        self.symptoms_time = self.infection_time + self.incubation_period

        if self.is_traced:
            self.isolation_time = self.symptoms_time    # traced cases are isolated with no delay
        elif self.is_subclinical:
            self.isolation_time = float('inf')      # subclinical cases are never isolated
        else:
            onset_to_isolation_delay = weibull_min.rvs(self.scenario.delay_shape, scale=self.scenario.delay_scale)
            self.isolation_time = self.symptoms_time + onset_to_isolation_delay

    def infect(self):
        number_of_new_cases = nbinom.rvs(n=self.scenario.R_0_disp, p=self.scenario.p)
        serial_intervals = skewnorm.rvs(
            self.scenario.serial_int_skewness,
            loc=self.incubation_period,
            scale=self.scenario.serial_int_scale,
            size=number_of_new_cases,
        )
        secondary_cases_times = [self.infection_time + serial_interval for serial_interval in serial_intervals]
        secondary_cases_times = [i for i in secondary_cases_times if i <= self.isolation_time and i <= self.scenario.T]   # ignore cases infected after T and after isolation
        # print(f"{len(secondary_cases_times)} new infections")
        for secondary_case_time in secondary_cases_times:
            person = Person(self.scenario, infection_time=secondary_case_time)
            self.scenario.new_case(person)


class Scenario:
    def __init__(
        self,
        T: float,
        T_control: float,
        initial_cases: int,
        rho: float,
        R_0: float,
        subclinical_prob: float,
        transmission_before_symptoms_percentage: int,
        onset_to_isolation: str,
    ):
        self.T = T
        self.T_control = T_control
        self.R_0 = R_0
        self.R_0_disp = 0.16    # Overdispersion in R_0
        self.p = self.R_0_disp / (self.R_0_disp + R_0)
        self.subclinical_prob = subclinical_prob
        self.rho = rho

        # Parameters for the Weibull distribution for generating incubation periods:
        self.incubation_p_shape = 2.322737  # shape (c / k)
        self.incubation_p_scale = 6.492272  # scale (lambda)

        # Parameters for the skew-normal distribution for generating serial intervals:
        self.serial_int_scale = 2
        match transmission_before_symptoms_percentage:
            case 1:
                self.serial_int_skewness = 30
            case 15:
                self.serial_int_skewness = 1.95
            case 30:
                self.serial_int_skewness = 0.7
            case _:
                raise ValueError("transmission_before_symptoms_percentage parameter must have value 1, 15 or 30.")

        # Parameters for the Weibull distribution for generating onset-to-isolation delay:
        match onset_to_isolation:
            case "short":
                self.delay_shape = 1.651524
                self.delay_scale = 4.287786
            case "long":
                self.delay_shape = 2.305172
                self.delay_scale = 9.483875
            case _:
                raise ValueError("onset_to_isolation parameter must have value 'short' or 'long'.")

        self.cases = []  # empty list to store all infected cases as Person objects
        self.queue = Queue()  # queue manages infections
        # Generate initial cases
        for _ in range(initial_cases):
            person = Person(self, is_traced=False, infection_time=0)
            self.new_case(person)
    
    
    def __repr__(self):
         return f"Scenario:\n\tT = {self.T}\n\trho = {self.rho}\n\tR_0 = {self.R_0}\n\tsubcl. prob. = {self.subclinical_prob}"


    def simulate(self):
        #print("Started", self)
        while not self.queue.empty():
            person = self.queue.get()
            person.infect()
        self.cases_in_control = sum([p.infection_time >= self.T_control for p in self.cases])
        #print(f"Simulation finished: {len(self.cases)} cases. Cases in control: {self.cases_in_control}.")

    def new_case(self, person: Person):
        self.queue.put(person)
        self.cases.append(person)

      

if __name__ == "__main__":
    scen = Scenario(
        T=20,
        T_control=15,
        initial_cases=5,
        rho=0.2,
        R_0=2.5,
        subclinical_prob=0.1,
        transmission_before_symptoms_percentage=1,
        onset_to_isolation="short",
    )
    scen.simulate()
