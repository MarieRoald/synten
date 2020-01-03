"""
Klassestruktur:

**Generator:**
Inneholder alle komponentene.

Input:
* Dimensjoner til bilde
* Antall tidssteg
* Overlappsrate (fra 0-1)
* Antall x-komponenter
* Antall y-komponenter

Attributter:
* Liste med bilder
* Liste med komponenter
* Dimensjoner til bilde
* Overlappsrate
* Antall x-komponenter
* Antall y-komponenter
* Faktormatrise (property)

Metoder:
* Generer komponenter


**Bilde:**
Inneholder kun komponenten og info om den.

Input:
* Lovlig område å tegne
* Dimensjoner til matrisa
* Antall tidssteg

Attributter:
* Tuppel av 2D matriser
* Lovlig område å tegne
* Antall tidssteg

**Komponent:**
Tar inn et bilde og tegner i den

Input:
* Radius parametre
* Shifteparametre
* Bilde

Attributter:
* Komponent-instans
* Komponent (property)

Metoder:
* Tegn tidssteg
* Flytt sentrum
* Blur
* Regn radius
* Tegn komponent
"""
from functools import partial
from itertools import product

import numpy as np
from numpy.polynomial import Polynomial
from skimage.draw import ellipse
from skimage.filters import gaussian as gaussian_blur

from .networks import BaseEvolvingComponentsGenerator


def _get_hyperrectangle(grids, idxes):
    start_point = tuple(grid[idx] for grid, idx in zip(grids, idxes))
    end_point = tuple(grid[idx+1] for grid, idx in zip(grids, idxes))
    return start_point, end_point


class ImageComponentGenerator(BaseEvolvingComponentsGenerator):
    """Componentgenerator for synthetic image factors
    """
    def __init__(
        self, 
        component_params,
        num_timesteps,
        image_shape,
        num_regions_per_axis,
        overlap=False,
        **kwargs
    ):
        assert len(component_params) <= np.prod(num_regions_per_axis)

        self.num_timesteps = num_timesteps
        self.image_shape = image_shape
        self.num_regions_per_axis = num_regions_per_axis
        self.overlap = overlap
        self.component_params = component_params

    def init_components(self):
        component_regions = self.generate_component_regions()
        self.images = [
            Image(
                shape=self.image_shape,
                component_region=component_region,
                num_timesteps=self.num_timesteps
            )
            for component_region in component_regions
        ]
        self.components = [
            ImageComponent(image, **kwargs) 
            for image, kwargs in zip(self.images, self.component_params)
        ]
        for component in self.components:
            component.init_component()
        
    def generate_factors(self):
        self.init_components()
        components = []
        for image, component in zip(self.images, self.components):
            component.draw_time_steps()
            component_timeseries = np.array(image.data)
            components.append(component_timeseries.reshape((len(component_timeseries), -1)))
                    
        return np.array(components).transpose(1, 2, 0)

    def generate_component_regions(self):
        """
        Returns list of rectangles parameterized by top-left and bottom-right coordinates
        Or equivalent in n dimensions.
        """
        if self.overlap:
            start_idxes = tuple(0 for _ in self.image_shape)
            end_idxes = tuple(length-1 for length in self.image_shape)
            return [(start_idxes, end_idxes) for _ in range(np.prod(self.num_regions_per_axis))]
        
        grids = [
            np.linspace(0, length-1, num_regions+1).astype(int)
            for length, num_regions in zip(self.image_shape, self.num_regions_per_axis)
        ]
        idxes = [range(num_regions) for num_regions in self.num_regions_per_axis]
        return [_get_hyperrectangle(grids, current_idxes) for current_idxes in product(*idxes)]



def sigmoidal(timestep, rate=1, carrying_capacity=1, offset_t=0, offset_r=0, inverted=False):
    timestep = timestep - offset_t
    radius = carrying_capacity/(1 + np.exp(-rate*carrying_capacity*timestep))

    if inverted:
        radius = carrying_capacity - radius
    
    return offset_r + radius


def linear(timestep, slope=1, intercept=0):
    return slope*timestep - intercept

def sinusiodal(timestep, angular_frequency=1, amplitude=1, offset=0, angular_offset=0):
    return amplitude*np.sin(timestep*angular_frequency + angular_offset) + offset


class ImageComponent:
    def __init__(self, image, radius_parameters, shift_parameters, blur_size=0):
        """
        shift_parameters : list[dict]
        """
        self.image = image
        self.radius_parameters = radius_parameters
        self.shift_parameters = shift_parameters
        self.blur_size = blur_size

    def _draw_time_step(self, position, radius, image_array):
        rr, cc = ellipse(*position, *radius, shape=image_array.shape, rotation=np.deg2rad(0))

        image_array[rr, cc] = 1
        if self.blur_size > 0:
            image_array = gaussian_blur(image_array, sigma=self.blur_size)
        return image_array

        
    def draw_time_steps(self):
        for timestep, time_slice in enumerate(self.image.data):
            position = self.get_position(timestep)
            radius = self.get_radius(timestep)
            time_slice[...] = self._draw_time_step(position, radius, time_slice)
        
    def calculate_positions(self, num_timesteps):
        self.positions = np.zeros(shape=(num_timesteps, 2))

        for axis in range(2):
            self.positions[0, axis] = self.shift_parameters[axis]['initial_position']
            for t in range(1, num_timesteps):
                perturbation = self.get_perturbation(axis)
                self.positions[t, axis] = self.positions[t-1, axis] + perturbation

    def init_component(self):
        self.calculate_positions(len(self.image))

    def get_radius(self, timestep):
        radius_type = self.radius_parameters['type']
        arguments = self.radius_parameters['arguments']
        if radius_type == 'quadratic':
            x_radius = Polynomial(**arguments['horisontal_coefficients'])
            y_radius = Polynomial(**arguments['vertical_coefficients'])
        elif radius_type == 'sigmoidal':
            x_radius = partial(sigmoidal, **arguments['horisontal_coefficients'])
            y_radius = partial(sigmoidal, **arguments['vertical_coefficients'])
        elif radius_type == 'linear':
            x_radius = partial(linear, **arguments['horisontal_coefficients'])
            y_radius = partial(linear, **arguments['vertical_coefficients'])
        else:
            pass
            
        return x_radius(timestep), y_radius(timestep)

    def get_position(self, timestep):
        return self.positions[timestep]

    def update_position(self):
        for axis in range(2):
            perturbation = self.get_perturbation(axis)
            self.position[axis] += perturbation
    
    def get_perturbation(self, axis):
        shift_probability = self.shift_parameters[axis]['shift_probability']
        speed = self.shift_parameters[axis].get('speed', 1)
        perturbation = 0
        if np.random.uniform() < shift_probability:
            increase_probability = 0.5 + 0.5*self.shift_parameters[axis].get('drift', 0)
            perturbation = speed*((np.random.uniform() < increase_probability)*2 - 1)
        
        return perturbation
        

class Image:
    """Wrapper for the images.

    Arguments
    ---------
    shape : tuple[int]
        Tuple with the shape of the full image
    component_region : tuple[tuple[int]]
        2-tuple with the coordinates of the minimum and maximum
        coordinate-corners of the box. For example,
        ((5, 2), (10, 8))
        parametrises a square that has the top left corner located
        at the point (5, 2) and bottom right corner located at
        (10, 8).
    num_timesteps : int
        The number of timesteps in the simulation.
    """
    def __init__(self, shape, component_region, num_timesteps):
        self.shape = shape
        self.component_region = component_region
        self.num_timesteps = num_timesteps
        self.data = tuple(np.zeros(shape) for _ in range(num_timesteps))
        
        self.component_mask = np.zeros(shape).astype(bool)
        slices = tuple(
            slice(start, stop, None) for start, stop in zip(*component_region)
        )
        self.component_mask[slices] = True

    def __len__(self):
        return len(self.data)