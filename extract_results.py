from pathlib import Path
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("experiments_folder", type=str)
    parser.add_argument("cp_evaluator_params", type=str)
    parser.add_argument("parafac2_evaluator_params", type=str)

    args = parser.parse_args()

    experiments_folder = args.experiments_folder
    experiments_folder = Path(experiments_folder)

    #with open(args.cp_evaluator_params) as f:
    #    cp_evaluator_params = json.load(f)
    #with open(args.parafac2_evaluator_params) as f:
    #    parafac2_evaluator_params = json.load(f)

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
         
            print('Calculate_metrics....')
            
            print(f"Evaluating {experiment}")
            if not (experiment / "summaries" / "summary.json").is_file():
                print(f"Skipping {experiment}")
                continue
            #evaluator.evaluate_experiment(str(experiment))


            print('Calculate_metrics....')
