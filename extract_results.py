from pathlib import Path
import argparse
import json
from tenkit_tools.evaluation.experiment_evaluator import ExperimentEvaluator
from csv import DictWriter

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


    num_rows = 0
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

            dataset_file = experiments_folder.parent/'datasets'/'_'.join(str(experiment_folder).split('_')[:-1])
            if "SeperateModeEvolvingFMS" in cp_evaluator_params['single_run_evaluator_params']:
                cp_evaluator_params["single_run_evaluator_params"]["SeperateModeEvolvingFMS"]["arguments"] = {
                    "evolving_tensor": str(dataset_file),
                    "internal_path": "evolving_tensor"
                }

            evaluator = ExperimentEvaluator(**cp_evaluator_params)
            folders = sorted(filter(lambda x: x.is_dir(), Path(experiment_folder).iterdir()))
            for i, experiment_subfolder in enumerate(folders):
                print(f"Evaluating {experiment_subfolder}")	
                if not (experiment_subfolder / "summaries" / "summary.json").is_file():
            	    print(f"Skipping {experiment_subfolder}")
            	    continue
                eval_results,_= evaluator.evaluate_experiment(str(experiment_subfolder),verbose=False)
                experiment_row = experiment.copy()
                experiment_row['attempt_num'] = i
                for metric in eval_results:
                    experiment_row.update(metric)
                print(experiment_row)

                open_mode = 'w' if num_rows == 0 else 'a'
                with (experiments_folder.parent/'results.csv').open(open_mode) as f:
                    writer = DictWriter(f, fieldnames=list(experiment_row.keys()))
                    if num_rows == '0':
                        writer.writeheader()
                    writer.writerow(experiment_row)
                    num_rows += 1

