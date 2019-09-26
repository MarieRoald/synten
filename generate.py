from synten.decomposition_generator import *


if __name__ == '__main__':
    shape = (40, 50, 60)
    rank_4_clusters =  {
        'clustering': {
                'num_informative_components': None,
                'class_sep': 2
        }
    }

    rank_4_networks = {
        'network': [
            'EvolvingNetworksGenerator',
            {
                'component_params': [
                    {
                        'init_size': 10,
                        'prob_adding': 0.01,
                        'prob_removing':10/shape[2]
                    },
                    {
                        'init_size': 10,
                        'prob_shifting': 1,
                        'prob_adding': 0,
                        'prob_removing': 0
                    },
                    {
                        'init_size': 3,
                        'prob_adding': (shape[1]//4-10)/shape[2],
                        'prob_removing': 0
                    },
                    {
                        'init_size': 8,
                        'prob_adding': (shape[1]//4-10)/shape[2],
                        'prob_removing': 0
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
        }
    ]
        },
    }
    generate_many_datasets(rank_4_clusters, rank_4_networks, rank_4_timeseries, shape, 4, 'jall', 20)
