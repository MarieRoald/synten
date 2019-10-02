import warnings
import h5py
from pathlib import Path
from tenkit.decomposition.decompositions import EvolvingTensor
from .timeseries import TimeSeriesGenerator
from .clusters import ClusterGenerator
from .networks import evolving_component_register
import numpy as np



def generate_evolving_tensor(
    clusters_params,
    networks_generator,
    networks_params,
    timeseries_params,
    shape,
    num_components,
    random_state=None
):
    cluster_generator = ClusterGenerator(
        generator_kwargs=dict(
            num_samples = shape[0],
            num_components=num_components,
            **clusters_params,
        ),
            random_state=random_state
    )
    # assert len(networks_params['component_params']) == num_components
    networks_generator = evolving_component_register[networks_generator](
        num_timesteps=shape[-1],
        num_nodes=shape[1],
        num_components=num_components,
        random_state=random_state,
        **networks_params,
    )
    assert len(timeseries_params['component_params']) == num_components
    timeseries_generator = TimeSeriesGenerator(num_timesteps=shape[-1], **timeseries_params, random_state=random_state)

    return EvolvingTensor(
        cluster_generator.generate_factors(),
        networks_generator.generate_factors(),
        timeseries_generator.generate_factors()
    ), cluster_generator.classes
    

def generate_dataset(
    clusters_params,
    networks_generator,
    networks_params,
    timeseries_params,
    shape,
    num_components,
    save_folder,
    name,
    random_state=None
):

    save_path = Path(save_folder)/name

    evolving_tensor, classes = generate_evolving_tensor(
        clusters_params,
        networks_generator,
        networks_params,
        timeseries_params,
        shape,
        num_components,
        random_state=random_state
    )
    tensor = np.array(evolving_tensor.construct_slices())

    with h5py.File(save_path,'w') as hdf5_file:
        evolving_tensor_group = hdf5_file.create_group('evolving_tensor')
        evolving_tensor.store_in_hdf5_group(evolving_tensor_group)

        data_tensor_group = hdf5_file.create_group('dataset')
        data_tensor_group['tensor'] = tensor
        data_tensor_group['classes'] = classes


def generate_many_datasets(
    clusters,
    networks,
    timeseries,
    shape,
    num_components,
    save_folder,
    num_datasets,
):
    for networks_name, network_info in networks.items():
        networks_generator, networks_params = network_info
        for clusters_name, clusters_params in clusters.items():
            for timeseries_name, timeseries_params in timeseries.items():
                for run in range(num_datasets):
                    random_state = np.random.RandomState(run)
                    name = f'{clusters_name}_{networks_name}_{timeseries_name}_{run}'
                    generate_dataset(
                        clusters_params,
                        networks_generator,
                        networks_params,
                        timeseries_params,
                        shape,
                        num_components,
                        save_folder,
                        name,
                        random_state=random_state
                    )


