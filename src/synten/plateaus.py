from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.utils import check_random_state
from .networks import BaseEvolvingComponentsGenerator
from subclass_register import SubclassRegister


parametrisation_register = SubclassRegister("Parametrisation")


@parametrisation_register.link_base
class BaseParametrisation:
    pass


class Constant(BaseParametrisation):
    def __init__(self, value):
        self.value = value

    def __call__(self, time):
        return self.value


class Sinusodial(BaseParametrisation):
    def __init__(self, offset, amplitude, angular_frequency=1, phase_angle=0):
        self.offset = offset
        self.amplitude = amplitude
        self.angular_frequency = angular_frequency
        self.phase_angle = phase_angle
    
    def __call__(self, time):
        return (
            self.offset
             + self.amplitude*np.sin(time*self.angular_frequency + self.phase_angle)
        )


def get_component_parametrisation(component_params):
    component_parametrisation = {}

    radius_type = component_params['radius_type']
    radius_params = component_params['radius_params']
    RadiusType = parametrisation_register[radius_type]
    component_parametrisation['radius'] = RadiusType(**radius_params)

    position_type = component_params['position_type']
    position_params = component_params['position_params']
    PositionType = parametrisation_register[position_type]
    component_parametrisation['position'] = PositionType(**position_params)

    return component_parametrisation


class PlateausNetworksGenerator(BaseEvolvingComponentsGenerator):
    def __init__(
        self,
        num_components,
        num_timesteps,
        num_nodes,
        component_params,
        smoothing_factor=0,
        offset=0,
        random_state=None
    ):
        self.num_components = num_components
        self.num_timesteps = num_timesteps
        self.num_nodes = num_nodes
        self.component_params = component_params
        self.smoothing_factor= smoothing_factor
        self.offset = offset
        self.random_state = random_state

    def init_components(self, rng):
        self.component_parametrisations = [
            get_component_parametrisation(component_params) for component_params in self.component_params
        ]

    def generate_factors(self):
        rng = check_random_state(self.random_state)
        self.init_components(rng)
        B = np.zeros(shape=(self.num_timesteps, self.num_nodes, self.num_components))

        for k, B_k in enumerate(B):
            for r, _ in enumerate(self.component_parametrisations):
                b_kr = self.make_component_timestep(r, k)

                if self.smoothing_factor > 0:
                    b_kr = gaussian_filter1d(b_kr, sigma=self.smoothing_factor, mode="nearest")
                
                B_k[:, r] = b_kr
        
        return B

    def make_component_timestep(self, component, timestep):
        radius = self.component_parametrisations[component]['radius'](timestep)
        radius = abs(radius)
        position = self.component_parametrisations[component]['position'](timestep)
        component_vector = np.zeros(self.num_nodes)

        min_pos = max(0, int(position - radius))
        max_pos = int(position + radius)

        component_vector[min_pos:max_pos] = 1
        return component_vector