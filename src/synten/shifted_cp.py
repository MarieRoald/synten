# lage nettverk ved å trekke tilfelidi uniform og så terskle det?

# et annet nettverk som gjør det samme og så smoother det etterpå?


from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.utils import check_random_state
from .networks import BaseEvolvingComponentsGenerator


class ShiftedCPNetworksGenerator(BaseEvolvingComponentsGenerator):
    def __init__(self, num_components, num_timesteps, num_nodes, smoothing_factor=0, offset=0, random_state=None):
        self.num_components = num_components
        self.num_timesteps = num_timesteps
        self.num_nodes = num_timesteps
        self.smoothing_factor= smoothing_factor
        self.offset = offset
        self.random_state = random_state

    def init_components(self, rng):
        self.B_0 = rng.normal(size=(self.num_nodes, self.num_components))
        self.B_0 = np.clip(self.B_0, a_min=0, a_max=None)
        self.B_0 /= np.linalg.norm(self.B_0, axis=0)
        self.B_0 = gaussian_filter1d(self.B_0, axis=0, sigma=self.smoothing_factor, mode="wrap")

    def generate_factors(self):
        rng = check_random_state(self.random_state)
        self.init_components(rng)
        B = np.empty(shape=(self.num_timesteps, self.num_nodes, self.num_components))

        for k, B_k in enumerate(B):
            B_k[...] = np.roll(self.B_0, axis=0, shift=k)
        
        return B
