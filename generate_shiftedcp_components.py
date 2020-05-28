from synten.decomposition_generator import *
from pathlib import Path
from synten import images
from itertools import product


if __name__ == '__main__':
    shape = (20, 30, 20)
    num_timesteps = 20
    num_nodes = 30
    num_subjects = 20



    rank_3_clusters =  {
        'clustering': {
                'num_informative_components': 3,
                'class_sep': 2
        }
    }

    rank_3_networks = {
        f'shifted_cp_smooth_{smoothing_factor}': [
            'ShiftedCPNetworksGenerator',
            {
                "smoothing_factor": smoothing_factor
            }
        ]

        for smoothing_factor in range(1, 5)
    }

    rank_3_timeseries = {
        'trends': {'component_params': [
            {
                'type': "RandomTimeComponent",
                'kwargs': {
                }
            },
            {
                'type': "RandomTimeComponent",
                'kwargs': {
                }
            },
            {
                'type': "RandomTimeComponent",
                'kwargs': {
                }
            },
        ]},
    }
    
    experiment_path = Path('shiftedcp_experiment_rank3/datasets')
    experiment_path.mkdir(parents=True, exist_ok=True)
    generate_many_datasets(rank_3_clusters, rank_3_networks, rank_3_timeseries, shape, 3, str(experiment_path), 5)
