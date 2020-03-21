from pathlib import Path
from itertools import product
import shutil
from tenkit_tools.experiment import Experiment
from tenkit.decomposition.block_parafac2 import RLS, Parafac2ADMM, BlockParafac2
from scipy.sparse import dok_matrix, csr_matrix
import numpy as np
# Skal denne fila være en del av synten?
# eller en del av tkt?  


# Mappestruktur
# Eksperiment
#  -> datasets
#     -> dataset1.h5
#     -> dataset2.h5
#     -> dataset2.h5
#  -> experiments


L = dok_matrix((25**2, 25**2))

for i, L_i in enumerate(L):
    if i - 25 >= 0:
        L[i, i-25] = -1
        L[i, i] += 1
    if i + 25 < L.shape[0]:
        L[i, i+25] = -1
        L[i, i] += 1
    if i - 1 >= 0:
        L[i, i-1] = -1
        L[i, i] = L[i, i] + 1
    if i + 1 < L.shape[0]:
        L[i, i+1] = -1
        L[i, i] = L[i, i] + 1

#L = csr_matrix(L)
L = np.zeros((30, 30))
for i, L_i in enumerate(L):
    if i > 0:
        L_i[i-1] -= 1
        L_i[i] += 1
    if i < 29:
        L_i[i+1] -= 1
        L_i[i] += 1


FLEXIBLE_DECOMPOSITION_PARAMS = {
    f'flexible_parafac2_coupling_{coupling}_smoothness_{smoothness}'.replace('.', '_'): {
        "type": "FlexibleParafac2_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 100,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, False, True],
            "coupling_strength": coupling,
            "init": "parafac2",
            "ridge_penalties": [smoothness*0.1, smoothness*0.1, smoothness*0.1],
            "tikhonov_matrices": [None, smoothness*L, None]
        }
    }
    for coupling, smoothness in product([30, 100, 300], [0.001, 0.33, 0.01])
}


FLEXIBLE_DECOMPOSITION_PARAMS = {
    f'flexible_parafac2_cohen_coupling_{coupling_strength}_max_coupling_{max_coupling}'.replace('.', '_'): {
        "type": "FlexibleParafac2_ALS",
        "arguments": {
            "max_its": 5000,
            "checkpoint_frequency": 1000,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, False, True],
            "coupling_strength": coupling_strength,
            "max_coupling": max_coupling,
            "init": "parafac2",
        }
    }
    for coupling_strength, max_coupling in product([1e-4, 1e-3, 1e-2, 1e-1, 1, 1.8e1, 3e1, 5.6e1, 100], [10, 1e2, 1e4, 1e6, 1e8, 1e10])
}

    #for coupling_strength, max_coupling in product([1e-4, 1e-3, 1e-2, 1e-1, 1, 1.8e1, 3e1, 5.6e1, 100], [10, 1e2, 1e4, 1e6, 1e8, 1e10])

ADMM_PARAFAC2_PARAMS = {
    f'parafac2_admm_smooth_{reg}': {
        "type": "BlockParafac2",
        "arguments": {
            "init": "random",
            "checkpoint_frequency": 500,
            "convergence_tol": 1e-8,
            "convergence_check_frequency": 10,
            "projection_update_frequency": 1,
            "max_its": 8000,
            "sub_problems": [
                RLS(mode=0, ridge_penalty=reg/10),
                Parafac2ADMM(rho=None, verbose=False, max_it=50, non_negativity=False, l2_similarity=L*reg),  # 0.2
                RLS(mode=2, non_negativity=True, ridge_penalty=reg/10),
            ]
        }
    } for reg in [1, 10, 100]
    #'parafac2_admm_0_2_50_sub_its': {
    #    "type": "BlockParafac2",
    #    "arguments": {
    #        "init": "random",
    #        "checkpoint_frequency": 500,
    #        "convergence_tol": 1e-8,
    #        "max_its": 8000,
    #        "sub_problems": [
    #            RLS(mode=0),
    #            Parafac2ADMM(rho=0.2, verbose=False, max_it=50),
    #            RLS(mode=2, non_negativity=True),
    #        ]
    #    }
    #},
    #'parafac2_admm_0_02_50_sub_its': {
    #    "type": "BlockParafac2",
    #    "arguments": {
    #        "init": "random",
    #        "checkpoint_frequency": 500,
    #        "convergence_tol": 1e-8,
    #        "max_its": 8000,
    #        "sub_problems": [
    #            RLS(mode=0),
    #            Parafac2ADMM(rho=0.02, verbose=False, max_it=50),
    #            RLS(mode=2, non_negativity=True),
    #        ]
    #    }
    #},

}

DECOMPOSITION_PARAMS = {
    **ADMM_PARAFAC2_PARAMS,
    'cp': {
        "type": "CP_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 500,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, False, True],
        }
    },
    'parafac2': {
        "type": "Parafac2_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 500,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, False, True],
            "print_frequency": -1,
        }
    },
}

DECOMPOSITION_PARAMS_OLD = {
    'flexible_parafac2_001': {
        "type": "FlexibleParafac2_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 100,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, True, True],
            "coupling_strength": 0.01,
            "init": "cp"
        }
    },
    'flexible_parafac2_01': {
        "type": "FlexibleParafac2_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 100,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, True, True],
            "coupling_strength": 0.1,
            "init": "cp"
        }
    },
    'flexible_parafac2_1': {
        "type": "FlexibleParafac2_ALS",
        "arguments": {
            "max_its": 8000,
            "checkpoint_frequency": 100,
            "convergence_tol": 1e-8,
            "non_negativity_constraints": [False, True, True],
            "coupling_strength": 1,
            "init": "cp"
        }
    },
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
    },
}
LOG_PARAMS = [
    {"type": "LossLogger"},
    {"type": "ExplainedVarianceLogger"},
    {"type": "CouplingErrorLogger", "arguments": {"not_flexible_ok": True}},
    {"type": "Parafac2ErrorLogger", "arguments": {"not_flexible_ok": True}},
]


def run_experiments(experiment_folder, rank, num_runs, noise_level, glob_pattern='*'):
    experiment_folder = Path(experiment_folder)
    for tensor_path in sorted((experiment_folder/'datasets/').glob(glob_pattern)):
        run_decompositions(tensor_path.name, experiment_folder, rank, num_runs, noise_level)
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
            "mode_names": ["time", "nodes", "samples"],
            "tensor_name": "dataset/tensor",
            "classes": [{}, {}, {"class": "dataset/classes"},]
        }
    }

def copy_best_run(experiment, save_path):
    """Copy the best-run checkpoint file from experiment to save_path"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best_run = Path(experiment.checkpoint_path)/experiment.summary['best_run']
    shutil.copy(best_run, save_path)


def run_decompositions(data_tensor_name, experiment_folder, rank, num_runs, noise_level):
    """Run num_runs CP and PARAFAC2 decomposition with the specified
    data tensor (filename) and experiment folder (path).
    """
    print(data_tensor_name)
    save_path = Path(experiment_folder)/'experiments'
    tensor_path = Path(experiment_folder)/'datasets'/data_tensor_name
    tensor_stem = Path(data_tensor_name).stem

    for decomposition in DECOMPOSITION_PARAMS:
        print(decomposition)
        experiment_params = {
            'save_path': f'{save_path}',
            'num_runs': num_runs,
            'experiment_name': f'{tensor_stem}_{decomposition}'
        }
        preprocessor_params = [
            #{
            #    "type": "GlobalScale",
            #    "arguments": {}
            #},
            {
                "type": "AddNoise",
                "arguments": {
                    "noise_level": noise_level
            }
            },
            {
                "type": "Transpose",
                "arguments": {"permutation": [1, 0, 2]}
            }
        ]
        decomposition_params = DECOMPOSITION_PARAMS[decomposition]
        decomposition_params['arguments']['rank'] = rank
        
        experiment = Experiment(
            experiment_params=experiment_params,
            data_reader_params=get_datareader_params(tensor_path),
            decomposition_params=decomposition_params,
            log_params=LOG_PARAMS,
            preprocessor_params=preprocessor_params,
            load_id=None,
        )
        experiment.run_experiments()

        best_run_filename = Path(experiment_folder)/'best_run'/f'{tensor_stem}_{decomposition}_noise_{noise_level}.h5'
        copy_best_run(experiment, best_run_filename)
