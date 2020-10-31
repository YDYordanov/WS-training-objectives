"""
Open results of an experiments and do summaries:
mean, std, convergence for:
WG-dev, WSC, KnowRef, DPR

Example usage:
python fill_final_table.py --dirs MAS_I_seeds MAS_II_seeds
"""

import os
import json
import argparse
import statistics
import numpy as np


def get_summaries(data_list):
    total_n = len(data_list)
    converging_data = [dat * 100 for dat in data_list
                       if dat >= 0.6]
    if converging_data != []:
        max_acc = max(converging_data)
        mean = sum(converging_data) / len(converging_data)
        stdev = statistics.stdev(converging_data)
        converging_seeds = len(converging_data)
        return_dict = {
            'total_number': total_n,
            'max': round(max_acc, 1),
            'mean': round(mean, 1),
            'stdev': round(stdev, 2),
            'converging_seeds': converging_seeds}
    else:
        print('No convergence!')
        return_dict = {
            'total_number': total_n,
            'max': None,
            'mean': None,
            'stdev': None,
            'converging_seeds': 0}
    if args.verbose:
        print('Converging data:', converging_data)

    return return_dict


def get_dir_list(dirs):
    all_directories = []
    for exper_dir in dirs:
        # List all subfolders of each of the directory list (dirs)
        all_folders = [
            dI for dI in os.listdir(exper_dir)
            if os.path.isdir(os.path.join(exper_dir, dI))
        ]
        all_directories += [os.path.join(exper_dir, folder)
                            for folder in all_folders]
    return all_directories


def collect_results(dirs, data_files):
    # First construct a file list of the given experiment
    all_directories = get_dir_list(dirs)

    # Open all results and get summaries
    print('Processing data...')
    all_data_lists = [{} for _ in data_files]

    for run_folder in all_directories:
        for idx, file_ in enumerate(data_files):
            file_path = os.path.join(run_folder, file_)
            with open(file_path, 'r') as f:
                data = json.load(f)
            for metric_name in data.keys():
                if metric_name not in all_data_lists[idx].keys():
                    all_data_lists[idx][metric_name] = []
                all_data_lists[idx][metric_name] += [data[metric_name]]

    return all_data_lists


def best_hyperparameters(dirs):
    exper_name = 'BWP_edited'
    # Extract the configuration dictionaries
    all_directories = get_dir_list(dirs)
    selected_dirs = all_directories
    config_dicts = []
    for run_dir in all_directories:
        config_file = os.path.join(run_dir, 'config.txt')
        if os.path.exists(config_file):
            with open(config_file) as json_file:
                config_dict = json.load(json_file)
                config_dicts.append(config_dict)

    class MyError(Exception):
        pass

    if selected_dirs == []:
        raise MyError("No runs found for experiment {}".format(exper_name))

    """
    Then we identify altered hyperparameters in cofig_dicts,
    and log their values: they can be columns of the table
    """

    column_dict_values = {}
    for key in config_dicts[0].keys():
        values_set = set([
            config_dict[key] for config_dict in config_dicts
        ])
        if len(values_set) > 1:
            column_dict_values[key] = list(values_set)

    if 'save_dir' in column_dict_values.keys():
        column_dict_values.pop('save_dir', None)
    if 'grad_accum_steps' in column_dict_values.keys():
        column_dict_values.pop('grad_accum_steps', None)
    if column_dict_values == {}:
        raise MyError("No columns found for experiment {} ".format(exper_name))

    # Extract the results of the experiment:
    all_result_dicts = []
    for run_dir in selected_dirs:
        config_file = os.path.join(run_dir, 'final_result.json')
        if not os.path.exists(config_file):
            config_file = os.path.join(run_dir, 'metrics.json')

        with open(config_file) as json_file:
            all_result_dicts.append(json.load(json_file))

    metrics = list(all_result_dicts[0].keys())
    if len(metrics) > 1:
        print('Available metrics for reporting:', metrics)
        print('Please type one of the given metrics...')
        metric = input()
    else:
        metric = metrics
    # metric_results = [d[metric] for d in all_result_dicts]
    # print('Results:', metric_results, 'Size:', len(metric_results))

    possible_columns = [
        'num_epochs', 'b_size', 'lr', 'model_seed',
        'seed', 'num_train_epochs', 'per_gpu_train_batch_size',
        'gradient_accumulation_steps', 'learning_rate']

    columns = [k for k in column_dict_values.keys() if k in possible_columns]
    print('Hyperparameter space:', {k: column_dict_values[k]
                                     for k in columns})
    column_sizes = [len(column_dict_values[col]) for col in columns]
    # Values of the metric for all hyperparam. combinations:
    values_array = np.zeros(column_sizes, dtype=float)
    # for key in column_dict_values.keys():
    #    for

    # Iterate over the array and fill-in the results
    it = np.nditer(values_array, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        # Construct indices for each hyperparam.:
        col_value_dict = {}
        for idx, value in enumerate(it.multi_index):
            column = columns[idx]
            col_value_dict[column] = column_dict_values[column][value]
            # print(col_value_dict)
        value = None
        for config_dict, results_dict in zip(config_dicts, all_result_dicts):
            if all(config_dict[key] == col_value_dict[key]
                   for key in col_value_dict.keys()):
                value = results_dict[metric]

        values_array[it.multi_index] = value
        it.iternext()
    # print('Results:', values_array)

    # Now take argmax to find the best hyperparam-s
    argmax_ = np.unravel_index(np.argmax(values_array), values_array.shape)
    # print(argmax_)
    print('Best hyperparameters:')
    for value_idx, column in zip(argmax_, columns):
        print(column, column_dict_values[column][value_idx])
    print('Best acc:', values_array[argmax_])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dirs", default=None, nargs='+', required=True,
        help="The experiment directories to summarize, "
             "with runs in sub-folders")
    parser.add_argument(
        "--result_files", default=None, nargs='+', required=False,
        help="List of result files to be visualised, e.g. WSC.json")
    parser.add_argument(
        "--best_hyperparams", action='store_true',
        help="Get the best hyperparameters")
    parser.add_argument(
        "--verbose", '-v', action='store_true',
        help="print stuff")
    args = parser.parse_args()

    # Collect the results
    if args.result_files is not None:
        results = collect_results(args.dirs, args.result_files)
        # Get the summaries of the result_files
        for i, file_name in enumerate(args.result_files):
            print('\n Results in {}:'.format(file_name))
            if results[i] == {}:
                print('File {} not found'.format(file_name))
            else:
                for key in results[i].keys():
                    print('   ', key, ':', get_summaries(results[i][key]))
        print('')

    if args.best_hyperparams:
        best_hyperparameters(dirs=args.dirs)
