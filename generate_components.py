from synten.decomposition_generator import *
from pathlib import Path


if __name__ == '__main__':
    shape = (40, 160, 50)
    rank_4_clusters =  {
        'clustering': {
                'num_informative_components': 4,
                'class_sep': 2
        }
    }

    rank_4_networks = {
        'network': [
            'EvolvingNetworksGenerator',
            {
                'component_params': [
                    {
                        'init_size': shape[1]//4//2,
                        'prob_adding': shape[1]/10/shape[2],
                        'prob_removing':shape[1]/5/shape[2]
                    },
                    {
                        'init_size': 10,
                        'prob_shifting': 1,
                        'prob_adding': 0,
                        'prob_removing': 0
                    },
                    {
                        'init_size': 10,
                        'prob_adding': 0.8*(shape[1]//4-10)/shape[2],
                        'prob_removing': 0
                    },
                    {
                        'init_size': 5,
                        'prob_adding': (shape[1]//4-10)/shape[2],
                        'prob_removing': (shape[1]//4-10)/shape[2]//5,
                        'prob_shifting': 0.5
                    }
                ]
            },
        ],
        'random': [
            'RandomNetworkGenerator',
            {
            }
        ]
    }

    rank_4_timeseries = {
        'trends': {'component_params': [
        {
            'type': "RandomTimeComponent",
            'kwargs': {
                'low': 1.1,
                'high': 2
            }
        },    {
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
                'b': 1,
                'c': shape[2]/2,
                'd': 1
            }
        },
        {
            'type': "TrigTimeComponent",
            'kwargs': {
                'a': 1,
                'b': 1,
                'c': 0,
                'd': 2.1
            }
        },
        ]},


        'random': {
        'component_params': [
        {
            'type': "RandomTimeComponent",
            'kwargs': {
                'low': 1.1,
                'high': 2
            }
        },
        {
            'type': "RandomTimeComponent",
            'kwargs': {
                'low': 1.1,
                'high': 2
            }
        },
        {
            'type': "RandomTimeComponent",
            'kwargs': {
                'low': 1.1,
                'high': 2
            }
        },
        {
            'type': "RandomTimeComponent",
            'kwargs': {
                'low': 1.1,
                'high': 2
            }
        }
        ]
        },
    }
    
    experiment_path = Path('experiments_many_nodes/datasets')
    experiment_path.mkdir(parents=True, exist_ok=True)
    generate_many_datasets(rank_4_clusters, rank_4_networks, rank_4_timeseries, shape, 4, str(experiment_path), 20)
