import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from subclass_register import SubclassRegister
from sklearn.utils import check_random_state


time_factor_register = SubclassRegister('time component')


@time_factor_register.link_base
class BaseTimeComponent(ABC):
    @abstractmethod
    def init_time_series(self):
        pass

    @abstractmethod
    def evolve_one_step(self):
        pass

    def iter_factor(self, num_timesteps, continued=False):
        if not continued:
            self.init_time_series()

        for _ in range(num_timesteps):
            self.evolve_one_step()
            yield self.factor_value

    def generate_factor(self, num_timesteps):
        return np.array(list(self.iter_factor(num_timesteps)))


class LinearTimeComponent(BaseTimeComponent):
    def __init__(self, a=1, b=0, init=None, init_std=1, random_state=None):
        self.a = a
        self.b = b
        self.init = init
        self.init_std = init_std
        self.random_state = random_state
        
    def init_time_series(self):
        rng = check_random_state(self.random_state)

        if self.init is None:
            init = self.init_std*rng.standard_normal() + self.b

        self.factor_value = init

    def evolve_one_step(self):
        self.factor_value = self.factor_value + self.a


class RandomTimeComponent(BaseTimeComponent):
    def __init__(self, low=0.1, high=1, random_state=None):
        self.low = low
        self.high = high
        self.random_state = random_state

    def init_time_series(self):
        self.rng_ = check_random_state(self.random_state)
        self.factor_value = self.rng_.uniform(self.low, self.high)

    def evolve_one_step(self):
        self.factor_value = self.rng_.uniform(high=self.high, low=self.low)
    

class ExponentialTimeComponent(BaseTimeComponent):
        def __init__(self, a=1, b=1, c=0, **kwargs):
            self.a = a
            self.b = b
            self.c = c
        
        def init_time_series(self):
            self.t = 0

        def evolve_one_step(self):
            self.t += 1
        
        @property
        def factor_value(self):
            return self.a*np.exp(self.b*self.t) + self.c


class LogisticTimeComponent(BaseTimeComponent):
    r""" Generate a factor according to
    .. math::
        \frac{a}{1 + e^{-bt + c}} + d
    """
    def __init__(self, a=1, b=1, c=10, d=0.1, **kwargs):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def init_time_series(self):
        self.t = 0

    def evolve_one_step(self):
        self.t += 1
    
    @property
    def factor_value(self):
        return self.a/(1+np.exp(-self.b*self.t + self.c)) + self.d


class LogisticWithExtictionTimeComponent(BaseTimeComponent):
    r""" Generate a factor by solving a logistic-like ODE

    ODE for :math:`t < extinction`:
    .. math::
        \frac{dx}{dt} = rx\left(1 - \frac{x}{K}\right)
    
    The ODE is flipped if :math:`t > extinction` so the carrying
    capacity is 0 and the unstable population is K.
    """
    def __init__(
        self,
        carrying_capacity=1,
        rate=0.5,
        start_extinct=False,
        flip_extinction=None,
        inverse_population=False,
        base=0,
        **kwargs,
    ):
        self.carrying_capacity = carrying_capacity
        self.rate = rate
        self.inverse_population = inverse_population
        self.extinct = start_extinct
        self.flip_extinction = flip_extinction
        self.base = base
    
    def init_time_series(self):
        self.t = 0
        self.hidden_state = 0.01
        if self.inverse_population:
            self.hidden_state = 0.99
    
    def evolve_one_step(self):
        self.t += 1
        if self.flip_extinction is not None and self.t in self.flip_extinction:
            self.extinct = not self.extinct

        if not self.extinct:
            self.hidden_state += self.rate*self.hidden_state*(1 - self.hidden_state)

        else:
            self.hidden_state -= self.rate*self.hidden_state*(1 - self.hidden_state)
        
        if self.hidden_state < 0.01:
            self.hidden_state = 0.01
        if self.hidden_state > 0.99:
            self.hidden_state = 0.99
        
        return self.hidden_state
    
    @property
    def factor_value(self):
        if self.inverse_population:
            return self.base + self.carrying_capacity*(1 - self.hidden_state)
        return self.base + self.carrying_capacity*self.hidden_state
    

class TrigTimeComponent(BaseTimeComponent):
    r""" Generate a factor according to
    .. math::
        a sin(b t + c) + d

    """
    def __init__(self, a=1, b=1, c=0, d=0, **kwargs):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def init_time_series(self):
        self.t = 0

    def evolve_one_step(self):
        self.t += 1
    
    @property
    def factor_value(self):
        return self.a*np.sin(self.b*self.t + self.c) + self.d


class TimeSeriesGenerator:
    def __init__(self, num_timesteps, component_params, random_state=None):
        """
        component_params : [
            {
                "type": "ExponentialTimeFactor",
                "kwargs": {"a": 1, "b": 2, "c": 0}
            },
            ...
        ]
        """
        self.component_params = component_params
        self.num_timesteps = num_timesteps
        self.random_state = random_state

    def generate_factors(self):
        factors = []
        rng = check_random_state(self.random_state)
        for component_parameters in self.component_params:
            component_name = component_parameters["type"]
            component_kwargs = component_parameters.get("kwargs", {})
            component_generator = time_factor_register[component_name](random_state=rng, **component_kwargs)

            factors.append(component_generator.generate_factor(self.num_timesteps))
        return np.array(factors).T
