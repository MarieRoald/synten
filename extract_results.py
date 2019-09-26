from pathlib import Path
import argparse

if __name__ == '__main__':
    parser = argparse.ArgmentParser()
    parser.add_argument("experiments_folder", type=str)
    args = parser.parse_args()


    experiments_folder = args["experiments_folder"]
    experiments_folder = Path(experiments_folder)
    for experiment_folder in experiments_folder.glob('*'):
            A_setup, B_setup, C_setup, dataset_num, model = experiment_folder.stem.split('_')

            experiment = {}
            experiment['experiment_name'] = experiment_folder.step
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