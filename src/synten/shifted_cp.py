# lage nettverk ved å trekke tilfelidi uniform og så terskle det?

# et annet nettverk som gjør det samme og så smoother det etterpå?


from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.utils import check_random_state
from .networks import BaseEvolvingComponentsGenerator


class PiecewiseConstantShiftedCPNetworksGenerator(BaseEvolvingComponentsGenerator):
    def __init__(
        self,
        num_components,
        num_timesteps,
        num_nodes,
        min_jumps=4,
        max_jumps=6,
        random_state=None
    ):
        if num_nodes % 2 != 0:
            raise ValueError("This generator mirrors the node-axis and therefore needs an even number of nodes.")
        if min_jumps % 2 != 0:
            raise ValueError("Need even number of jumps")
        if max_jumps % 2 != 0:
            raise ValueError("Need even number of jumps")

        self.num_components = num_components
        self.num_timesteps = num_timesteps
        self.num_nodes = num_nodes
        self.min_jumps = min_jumps
        self.max_jumps = max_jumps
        self.random_state = random_state
    
    def generate_factor_vector(self, rng):
        num_nodes = self.num_nodes
        min_jumps = self.min_jumps // 2
        max_jumps = self.max_jumps // 2
        num_jumps = np.random.randint(min_jumps, max_jumps+1)

        # Generate a sparse "derivative" vector
        factor_derivative = np.zeros(num_nodes//2)
        jump_values = rng.standard_normal(num_nodes//2)
        jump_indexes = rng.permutation(num_nodes//2)[:num_jumps]
        factor_derivative[jump_indexes] = jump_values[jump_indexes]

        # Integrate the sparse derivative vector to obtain a piecewise constant vector
        factor_vector = np.cumsum(factor_derivative)

        # Flip it to obtain a periodic vector
        factor_vector = np.concatenate([factor_vector, np.flip(factor_vector)])
        return factor_vector

    def init_components(self, rng):
        self.B_0 = np.zeros(shape=(self.num_nodes, self.num_components))
        for component in range(self.num_components):
            self.B_0[:, component] = self.generate_factor_vector(rng)

    def generate_factors(self):
        rng = check_random_state(self.random_state)
        self.init_components(rng)
        B = np.zeros(shape=(self.num_timesteps, self.num_nodes, self.num_components))
        for k, B_k in enumerate(B):
            B_k[...] = np.roll(self.B_0, axis=0, shift=k)
        return B


class ShiftedCPNetworksGenerator(BaseEvolvingComponentsGenerator):
    def __init__(
        self,
        num_components,
        num_timesteps,
        num_nodes,
        smoothing_factor=0,
        random_state=None
    ):
        self.num_components = num_components
        self.num_timesteps = num_timesteps
        self.num_nodes = num_nodes
        self.smoothing_factor= smoothing_factor
        self.random_state = random_state

    def init_components(self, rng):
        self.B_0 = rng.normal(size=(self.num_nodes, self.num_components))
        self.B_0 = np.clip(self.B_0, a_min=0, a_max=None)
        self.B_0 /= np.linalg.norm(self.B_0, axis=0)
        if self.smoothing_factor > 0:
            self.B_0 = gaussian_filter1d(self.B_0, axis=0, sigma=self.smoothing_factor, mode="wrap")

    def generate_factors(self):
        rng = check_random_state(self.random_state)
        self.init_components(rng)
        B = np.empty(shape=(self.num_timesteps, self.num_nodes, self.num_components))

        for k, B_k in enumerate(B):
            B_k[...] = np.roll(self.B_0, axis=0, shift=k)
        
        return B
