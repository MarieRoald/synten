from synten.decomposition_generator import *
from pathlib import Path
from synten import images
from itertools import product


if __name__ == '__main__':
    shape = (40, 160, 50)
    num_timesteps = 40
    num_nodes = 160
    num_subjects = 50



    rank_2_clusters =  {
        'clustering': {
                'num_informative_components': 2,
                'class_sep': 2
        }
    }

    rank_2_networks = {
        f'shifted_cp_smooth_{smoothing_factor}': [
            'ShiftedCPNetworksGenerator',
            {
                "smoothing_factor": smoothing_factor
            }
        ]

        for smoothing_factor in range(1, 5)
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
    
    experiment_path = Path('shiftedcp_experiment/datasets')
    experiment_path.mkdir(parents=True, exist_ok=True)
    generate_many_datasets(rank_2_clusters, rank_2_networks, rank_2_timeseries, shape, 2, str(experiment_path), 5)
