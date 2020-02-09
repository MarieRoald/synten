from synten.decomposition_generator import *
from pathlib import Path
from synten import images
from itertools import product


def generate_sigmoid_images(
    num_timesteps = 5,
    image_shape = (20, 20),
    min_r = 1,
    max_r = 3,
    rate = 0.5,
    #all_offset_ts = [.10, .35, .65, .8],
    all_offset_ts = [0.33, 0.66],
    #all_inverted = [False, False, True, True],
    all_inverted = [False, True],
    #all_positions = list(itertools.product((int(image_shape[0]/4), int(3*image_shape[0]/4)), repeat=2)),
    all_positions = [(1/3, 1/3), (2/3, 2/3)],
    shift_probability = 0.1,
    speed = 1,
    blur_size=1
):
    carrying_capacity = max_r - min_r
    return {
        'image_shape': image_shape,
        'num_regions_per_axis': (2, 1),
        'overlap': True,
        'component_params': [
                    {
                'blur_size': blur_size,
                'radius_parameters': {
                    'type': 'sigmoidal',
                    'arguments': {
                        'horisontal_coefficients': {
                            'carrying_capacity': carrying_capacity,
                            'offset_r': min_r,
                            'offset_t': offset_t*num_timesteps,
                            'rate': rate,
                            'inverted': inverted,
                        },
                        'vertical_coefficients': {
                            'carrying_capacity': carrying_capacity,
                            'offset_r': min_r,
                            'offset_t': offset_t*num_timesteps,
                            'rate': rate,
                            'inverted': inverted,
                        }
                    }
                },
                'shift_parameters': [
                    {
                        'speed': speed,
                        'shift_probability': shift_probability,
                        'initial_position': int(position[0]*image_shape[0]),
                    },
                    {
                        'speed': speed,
                        'shift_probability': shift_probability,
                        'initial_position': int(position[1]*image_shape[1]),
                    },
                ]
            }

            for offset_t, inverted, position in zip(all_offset_ts, all_inverted, all_positions)
        ]
    }


if __name__ == '__main__':
    # shape = (40, 160, 50)
    num_timesteps = 5
    image_shape = (25, 25)
    num_subjects = 40
    shape = (num_subjects, np.prod(image_shape), num_timesteps)
    shift_probabilities = [0.5, 0.75, 1]
    speeds = [1, 2, 4]



    rank_2_clusters =  {
        'clustering': {
                'num_informative_components': 2,
                'class_sep': 2
        }
    }

    rank_2_networks = {
        f'image_shift_{shift_probability}_speed_{speed}'.replace('.', '_'): [
            'ImageComponentGenerator',
            generate_sigmoid_images(num_timesteps, image_shape, shift_probability=shift_probability, speed=speed),
        ]

        for shift_probability, speed in product(shift_probabilities, speeds)
    }

    rank_2_timeseries = {
        'trends': {'component_params': [
            {
                'type': "ExponentialTimeComponent",
                'kwargs': {
                    'a': 1,
                    'b': -0.2,
                    'c': 1
                }
            },
            {
                'type': "LogisticTimeComponent",
                'kwargs': {
                    'a': 2,
                    'b': 2,
                    'c': num_timesteps/2,
                    'd': 1
                }
            },
        ]},
    }
    
    experiment_path = Path('smooth_experiments_first_informed_test/datasets')
    experiment_path.mkdir(parents=True, exist_ok=True)
    generate_many_datasets(rank_2_clusters, rank_2_networks, rank_2_timeseries, shape, 2, str(experiment_path), 3)
