from pathlib import Path
import shutil
from tenkit_tools.experiment import Experiment
# Skal denne fila være en del av synten?
# eller en del av tkt?  


# Mappestruktur
# Eksperiment
#  -> datasets
#     -> dataset1.h5
#     -> dataset2.h5
#     -> dataset2.h5
#  -> experiments



DECOMPOSITION_PARAMS = {
    'cp': {
        "type": "CP_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 100,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, False, True],
        }
    },
    'parafac2': {
        "type": "Parafac2_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 100,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, False, True],
            "print_frequency": -1,
        }
    }
}
LOG_PARAMS = [
    {"type": "LossLogger"},
    {"type": "ExplainedVarianceLogger"}
]


def run_experiments(experiment_folder, rank, num_runs):
    experiment_folder = Path(experiment_folder)
    for tensor_path in sorted((experiment_folder/'datasets/').iterdir()):
        run_decompositions(tensor_path.name, experiment_folder, rank, num_runs)
    # hente num_runs og tol fra experiments_parameters
    # Er det å ikke bruke tkt å finne opp hjulet?
    # iterer over alle dataset i mappa (og undermapper?)
    # - les inn dataset
    # - run_decomposition(data, rank, num_runs)


def get_datareader_params(tensor_path):
    return {
        "type": "HDF5DataReader",
        "arguments": {
            "file_path": str(tensor_path),
            "meta_info_path": str(tensor_path),
            "mode_names": ["samples", "nodes", "time"],
            "tensor_name": "dataset/tensor",
            "classes": [{"class": "dataset/classes"}, {}, {}]
        }
    }

def copy_best_run(experiment, save_path):
    """Copy the best-run checkpoint file from experiment to save_path"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_run = Path(experiment.checkpoint_path)/experiment.summary['best_run']
    shutil.copy(best_run, save_path)


def run_decompositions(data_tensor_name, experiment_folder, rank, num_runs):
    """Run num_runs CP and PARAFAC2 decomposition with the specified
    data tensor (filename) and experiment folder (path).
    """
    print(data_tensor_name)
    save_path = Path(experiment_folder)/'experiments'
    tensor_path = Path(experiment_folder)/'datasets'/data_tensor_name
    tensor_stem = Path(data_tensor_name).stem

    for decomposition in ['cp', 'parafac2']:
        print(decomposition)
        experiment_params = {
            'save_path': f'{save_path}',
            'num_runs': num_runs,
            'experiment_name': f'{tensor_stem}_{decomposition}'
        }
        decomposition_params = DECOMPOSITION_PARAMS[decomposition]
        decomposition_params['arguments']['rank'] = rank
        
        experiment = Experiment(
            experiment_params=experiment_params,
            data_reader_params=get_datareader_params(tensor_path),
            decomposition_params=decomposition_params,
            log_params=LOG_PARAMS,
            preprocessor_params=[],
            load_id=None,
        )
        experiment.run_experiments()

        best_run_filename = Path(experiment_folder)/'best_run'/f'{tensor_stem}_{decomposition}.h5'
        copy_best_run(experiment, best_run_filename)