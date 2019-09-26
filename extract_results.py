from pathlib import Path
import argparse
import json
from tenkit_tools.evaluation.experiment_evaluator import ExperimentEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_folder", type=str)
    parser.add_argument("cp_evaluator_params", type=str)
    parser.add_argument("parafac2_evaluator_params", type=str)

    args = parser.parse_args()

    experiments_folder = Path(args.experiments_folder)

    with open(args.cp_evaluator_params) as f:
        cp_evaluator_params = json.load(f)
    with open(args.parafac2_evaluator_params) as f:
        parafac2_evaluator_params = json.load(f)



    for experiment_folder in sorted(experiments_folder.glob('*')):
            A_setup, B_setup, C_setup, dataset_num, model = experiment_folder.stem.split('_')

            experiment = {}
            experiment['experiment_name'] = experiment_folder.stem
            experiment['A_setup'] = A_setup
            experiment['B_setup'] = B_setup
            experiment['C_setup'] = C_setup
            experiment['dataset_num'] = dataset_num
            experiment['model'] = model

            print(f'Loading experiment: {experiment["experiment_name"]}')
            print(f'    A setup: {A_setup}')
            print(f'    B setup: {B_setup}')
            print(f'    C setup: {C_setup}')
            print(f'    Dataset number: {dataset_num}')
            print(f'    Model: {model}')
            evaluator = ExperimentEvaluator(**cp_evaluator_params)
            
            for experiment_subfolder in sorted(filter(lambda x: x.is_dir(), Path(experiment_folder).iterdir())):
            	print(f"Evaluating {experiment_subfolder}")	
            	if not (experiment_subfolder / "summaries" / "summary.json").is_file():
            	    print(f"Skipping {experiment_subfolder}")
            	    continue
		
            	evaluator.evaluate_experiment(str(experiment_subfolder))

