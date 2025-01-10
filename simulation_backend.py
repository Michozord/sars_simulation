# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 12:25:10 2024

@author: Michal Trojanowski
"""

from __future__ import annotations

from scipy.stats import bernoulli, nbinom, uniform, weibull_min, skewnorm
from scipy.optimize import minimize
from queue import Queue
from dataclasses import dataclass
from joblib import Parallel, delayed
from typing import List
import numpy as np


class Person:
    def __init__(
        self,
        simulation: Simulation,
        infection_time: float,
        is_traced: bool = None,
    ):
        self.simulation = simulation
        # Determine timeline for this person:
        #
        #   TIMELINE:
        #   -- x ------------- x ------------------ x --------->
        #   infection   |   symptoms    |       isolation
        #               |               |
        #         incubation      onset to isolation delay
        #
        if is_traced is None:
            self.is_traced = bool(bernoulli.rvs(self.simulation.scenario_parameters.rho))
        else:
            self.is_traced = is_traced
        self.is_subclinical = bool(bernoulli.rvs(self.simulation.scenario_parameters.subclinical_prob))
        self.infection_time = infection_time 
        self.incubation_period = weibull_min.rvs(
            self.simulation.scenario_parameters.incubation_p_shape, 
            scale=self.simulation.scenario_parameters.incubation_p_scale
        )
        self.symptoms_time = self.infection_time + self.incubation_period

        if self.is_traced:
            self.isolation_time = self.symptoms_time    # traced cases are isolated with no delay
        elif self.is_subclinical:
            self.isolation_time = float('inf')      # subclinical cases are never isolated
        else:
            onset_to_isolation_delay = weibull_min.rvs(
                self.simulation.scenario_parameters.delay_shape, 
                scale=self.simulation.scenario_parameters.delay_scale
            )
            self.isolation_time = self.symptoms_time + onset_to_isolation_delay

    def infect(self):
        number_of_new_cases = nbinom.rvs(n=self.simulation.scenario_parameters.R_0_disp, p=self.simulation.scenario_parameters.p)
        serial_intervals = skewnorm.rvs(
            self.simulation.scenario_parameters.serial_int_skewness,
            loc=self.incubation_period,
            scale=self.simulation.scenario_parameters.serial_int_scale,
            size=number_of_new_cases,
        )
        secondary_cases_times = [self.infection_time + serial_interval for serial_interval in serial_intervals]
        secondary_cases_times = [i for i in secondary_cases_times if i <= self.isolation_time]   # ignore cases infected after isolation
        
        for secondary_case_time in secondary_cases_times:
            if secondary_case_time > self.simulation.scenario_parameters.T:     # skip infections after time T
                continue
            person = Person(self.simulation, infection_time=secondary_case_time)
            self.simulation.new_case(person)

        return len(secondary_cases_times)


@dataclass
class ScenarioParameters:
    T: float
    T_control: float
    initial_cases: int
    rho: float
    R_0: float
    subclinical_prob: float
    transmission_before_symptoms_percentage: int
    onset_to_isolation: str

    serial_int_scale: float = 2

    R_0_disp: float = 0.16

    incubation_p_shape: float = 2.322737
    incubation_p_scale: float = 6.492272

    def __post_init__(self):
        self.p = self.R_0_disp / (self.R_0_disp + self.R_0)

        match self.transmission_before_symptoms_percentage:
            case 1:
                self.serial_int_skewness = 30
            case 15:
                self.serial_int_skewness = 1.95
            case 30:
                self.serial_int_skewness = 0.7
            case _:
                raise ValueError("transmission_before_symptoms_percentage parameter must have value 1, 15 or 30.")
            
        match self.onset_to_isolation:
            case "short":
                self.delay_shape = 1.651524
                self.delay_scale = 4.287786
            case "long":
                self.delay_shape = 2.305172
                self.delay_scale = 9.483875
            case _:
                raise ValueError("onset_to_isolation parameter must have value 'short' or 'long'.")


class Simulation:
    def __init__(
        self,
        scenario_parameters: ScenarioParameters
    ):
        self.scenario_parameters = scenario_parameters
        self.cases = []  # empty list to store all infected cases as Person objects
        self.queue = Queue()  # queue manages infections

        self.effective_R_0_vector = []

        # Generate initial cases
        for _ in range(self.scenario_parameters.initial_cases):
            person = Person(self, is_traced=False, infection_time=0)
            self.new_case(person)
    
    
    def __repr__(self):
         return f"Scenario:\n\tT = {self.scenario_parameters.T}\n\trho = {self.scenario_parameters.rho}\n\tR_0 = {self.scenario_parameters.R_0}\n\tsubcl. prob. = {self.scenario_parameters.subclinical_prob}"


    def simulate(self):
        while not self.queue.empty() and len(self.cases) < 5000:
            person = self.queue.get()
            number_of_new_cases = person.infect()
            self.effective_R_0_vector.append(number_of_new_cases)

        self.cases_in_control = sum([p.infection_time >= self.scenario_parameters.T_control for p in self.cases])
        if len(self.cases) >= 5000 and self.cases_in_control == 0:
            self.cases_in_control += 1

    def new_case(self, person: Person):
        self.queue.put(person)
        self.cases.append(person)


@dataclass
class SimulationResults:
    effective_R_0: float
    is_controlled: bool

@dataclass
class ScenarioStatistics:
    effective_R_0_median: float
    controlled_percentage: float


class Scenario:
    def __init__(self, scenario_parameters: ScenarioParameters):
        self.scenario_parameters = scenario_parameters

    def _run_single(self, scenario_parameters: ScenarioParameters):
        simulation = Simulation(scenario_parameters)
        simulation.simulate()
        return SimulationResults(
            np.mean(simulation.effective_R_0_vector),
            simulation.cases_in_control == 0
        )
        

    def run_simulations(self, num_simulations=1000):
        results = Parallel(n_jobs=-1, backend='loky', verbose=0)(
            delayed(self._run_single)(self.scenario_parameters) for _ in range(num_simulations)
        )

        return ScenarioStatistics(
            effective_R_0_median=np.median([sim_results.effective_R_0 for sim_results in results]),
            controlled_percentage=np.sum([sim_results.is_controlled for sim_results in results]) / num_simulations
        )

