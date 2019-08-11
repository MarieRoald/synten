import numpy as np
from abc import ABC, abstractmethod


class BaseTimeFactor(ABC):
    @abstractmethod
    def init_time_series(self):
        pass

    @abstractmethod
    def evolve_one_step(self):
        pass

    @abstractmethod
    def get_factor_value(self):
        pass


class LinearTimeFactor(BaseTimeFactor):
    def init_time_series(self, a=1, b=0, init=None, init_std=1,):
        if init is None:
            init = init_std*np.random.standard_normal() + b

        self.current_component = init
        self.a = a


    def evolve_one_step(self):
        self.current_component = self.current_component + self.a


    def get_factor_value(self):
        return self.current_component



class RandomTimeFactor(BaseTimeFactor):
    def init_time_series(self, low=0.1, high=1):
        self.low = low
        self.high = high

        self.current_component = np.random.uniform(low, high)


    def evolve_one_step(self):
        self.current_component = np.random.uniform(high=self.high, low=self.low)
    
    def get_factor_value(self):
        return self.current_component


class ExponentialTimeFactor(LinearTimeFactor):

        def init_time_series(self, a=1, b=1, c = 0):
            self.a = a
            self.b = b
            self.c = c
        
            self.t = 0

        def evolve_one_step(self):
            self.t += 1
        
        @property
        def current_component(self):
            return self.a*np.exp(self.b*self.t) + self.c

        def get_factor_value(self):
            return self.current_component

class SigmoidTimeFactor(LinearTimeFactor):
        def init_time_series(self, a=1, b=1, c=10, d=0.1):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
        
            self.t = 0

        def evolve_one_step(self):
            self.t += 1
        
        @property
        def current_component(self):
            return self.a/(1+np.exp(-self.b*self.t + self.c)) + self.d

        def get_factor_value(self):
            return self.current_component

class LogisticWithExtiction:
    def init_time_series(
        self,
        carrying_capacity=1,
        rate=0.5,
        start_extinct=False,
        flip_extinction=None,
        inverse_population=False,
        base=0
    ):
        self.carrying_capacity = carrying_capacity
        self.rate = rate
        self.t = 0
        self.current_component = 0.01
        self.inverse_population = inverse_population
        if inverse_population:
            self.current_component = 0.99
        self.extinct = start_extinct
        self.flip_extinction = flip_extinction
        self.base = base

    
    def evolve_one_step(self):
        self.t += 1
        if self.flip_extinction is not None and self.t in self.flip_extinction:
            self.extinct = not self.extinct

        if not self.extinct:
            self.current_component += self.rate*self.current_component*(1 - self.current_component)

        else:
            self.current_component -= self.rate*self.current_component*(1 - self.current_component)
        
        if self.current_component < 0.01:
            self.current_component = 0.01
        if self.current_component > 0.99:
            self.current_component = 0.99
        
        return self.get_factor_value()
    
    def get_factor_value(self):
        if self.inverse_population:
            return self.base + self.carrying_capacity*(1 - self.current_component)
        return self.base + self.carrying_capacity*self.current_component
    


class TrigTimeFactor(LinearTimeFactor):
        def init_time_series(self, a=1, b=1, c=0, d=0):
            self.a = a
            self.b = b
            self.c = c
            self.d = d
        
            self.t = 0

        def evolve_one_step(self):
            self.t += 1
        
        @property
        def current_component(self):
            return self.a*np.sin(self.b*self.t + self.c) + self.d

        def get_factor_value(self):
            return self.current_component


class TimeSeriesFactorGenerator:
    def __init__(self, num_timesteps, generators, init_kwargs):
        """
        generators : [(ExponentialTimeFactor, {}), ...]
        init_kwargs = [{'a': 1, 'b': 2, 'c': 0}, ...]
        """
        self.generators = [Generator(**kwargs) for Generator, kwargs in generators]
        self.init_kwargs = init_kwargs
        self.num_timesteps = num_timesteps

    def generate_factors(self):
        for generator, init_kwarg in zip(self.generators, self.init_kwargs):
            generator.init_time_series(**init_kwarg)
        
        factors = []
        for generator in self.generators:
            component = []
            for i in range(self.num_timesteps):
                generator.evolve_one_step()
                component.append(generator.get_factor_value())
            factors.append(component)
        return np.array(factors).T
