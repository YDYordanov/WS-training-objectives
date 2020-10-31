"""
Experiment 2: Training WG-SR on 20 seeds
WG-SR is trained on additional 10 seeds, and is evaluated separately
"""

import argparse
from gpu_utils import *


parser = argparse.ArgumentParser(description="Transformers model")
parser.add_argument("--total_num_gpus", '-num_gpus', type=int, default=1,
                    help='Total number of GPUs in the system')
parser.add_argument("--ignore_gpu_ids", '-ignore_gpus', default=[], nargs='+',
                    help='The interval-separated list of ids of the GPUs '
                         'in the system not to be used for running '
                         '(to be ignored by this script).')
args = parser.parse_args()

synch_file_path = 'winogrande/gpu_synch.csv'
# Create an empty gpu_synch file for GPU synchronization
prep_gpus(synch_file_path, taken_gpus_list=args.ignore_gpu_ids,
          num_gpus=args.total_num_gpus)

""" WG-SR Seeds """

exper_name = 'WG-SR_seeds'
folder_number = 1
per_gpu_train_batch_size = 8

for lr in [1e-5]:
    for gradient_accumulation_steps in [2]:
        for ep in [5]:
            for seed in [
                43289, 5689, 89304, 868843, 87, 43226, 965, 109, 89324, 679,
                26499, 12924, 87013, 77215, 94508, 44228, 78142, 13704, 57428,
                73493
            ]:
                # Wait for a GPU to free-up and find its idx
                gpu_idx = find_unassigned_gpu(synch_file_path)
                update_gpu_synch_file(synch_file_path, gpu_idx,
                                      is_running=True)

                command = "nohup python winogrande/scripts/run_experiment.py" \
                          " --model_type roberta_mc " \
                          "--model_name_or_path roberta-large " \
                          "--task_name winogrande --do_eval " \
                          "--data_dir winogrande/Data --max_seq_length 80 " \
                          "--per_gpu_eval_batch_size 4 " \
                          "--per_gpu_train_batch_size {} " \
                          "--gradient_accumulation_steps={} " \
                          "--learning_rate={} --num_train_epochs={} " \
                          "--output_dir ./saved_models/{}/{} " \
                          "--do_train --logging_steps 4744 " \
                          "--save_steps 6000 --seed={} " \
                          "--data_cache_dir ./output/cache/ " \
                          "--warmup_pct 0.1 --evaluate_during_training " \
                          "--exper_name={} --use_devices={} " \
                          "> logs/{}_{}.out &".format(
                            per_gpu_train_batch_size,
                            gradient_accumulation_steps, lr, ep,
                            exper_name, folder_number, seed, exper_name,
                            gpu_idx, exper_name, folder_number)
                # Set the GPU to use
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print("Executing:", command)
                os.system(
                    'export PYTHONPATH=$PYTHONPATH:$(pwd)/winogrande ; '
                    + command)
                folder_number += 1

""" WG-SR Extra 10 seeds """

exper_name = 'WG-SR_extra_seeds'
folder_number = 1
per_gpu_train_batch_size = 8

for lr in [1e-5]:
    for gradient_accumulation_steps in [2]:
        for ep in [5]:
            for seed in [
                419899, 526171, 527830, 621227, 136638, 562451,
                597061, 302992, 394980, 381220
            ]:
                # Wait for a GPU to free-up and find its idx
                gpu_idx = find_unassigned_gpu(synch_file_path)
                update_gpu_synch_file(synch_file_path, gpu_idx,
                                      is_running=True)

                command = "nohup python winogrande/scripts/run_experiment.py" \
                          " --model_type roberta_mc " \
                          "--model_name_or_path roberta-large " \
                          "--task_name winogrande --do_eval " \
                          "--data_dir winogrande/Data --max_seq_length 80 " \
                          "--per_gpu_eval_batch_size 4 " \
                          "--per_gpu_train_batch_size {} " \
                          "--gradient_accumulation_steps={} " \
                          "--learning_rate={} --num_train_epochs={} " \
                          "--output_dir ./saved_models/{}/{} " \
                          "--do_train --logging_steps 4744 " \
                          "--save_steps 6000 --seed={} " \
                          "--data_cache_dir ./output/cache/ " \
                          "--warmup_pct 0.1 --evaluate_during_training " \
                          "--exper_name={} --use_devices={} " \
                          "> logs/{}_{}.out &".format(
                            per_gpu_train_batch_size,
                            gradient_accumulation_steps, lr, ep,
                            exper_name, folder_number, seed, exper_name,
                            gpu_idx, exper_name, folder_number)
                # Set the GPU to use
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print("Executing:", command)
                os.system(
                    'export PYTHONPATH=$PYTHONPATH:$(pwd)/winogrande ; '
                    + command)
                folder_number += 1

""" WG-SR Seeds Evaluation """

exper_name = 'WG-SR_seeds_eval'
folder_number = 1
per_gpu_train_batch_size = 8

data_dir = "winogrande/saved_models/WG-SR_seeds".format(exper_name)
filenames = os.listdir(data_dir)
run_folders = []
for filename in filenames:
    if os.path.isdir(os.path.join(os.path.abspath(data_dir), filename)):
        run_folders.append(filename)
run_folders.sort()

for run_folder in run_folders:
    model_folder = os.path.join(data_dir, run_folder)
    for eval_dataset in ['WSC', 'DPR', 'WG-dev']:
        # Wait for a GPU to free-up and find its idx
        gpu_idx = find_unassigned_gpu(synch_file_path)
        update_gpu_synch_file(synch_file_path, gpu_idx,
                              is_running=True)

        command = "nohup python scripts/eval_experiment.py " \
                  "--model_type roberta_mc " \
                  "--model_name_or_path roberta-large " \
                  "--task_name winogrande --do_eval --do_lower_case " \
                  "--data_dir winogrande/Data/{} --max_seq_length 80 " \
                  "--per_gpu_eval_batch_size 4 " \
                  "--per_gpu_train_batch_size 8 " \
                  "--gradient_accumulation_steps=2 " \
                  "--learning_rate 1e-5 --num_train_epochs 0 " \
                  "--output_dir={} --logging_steps 4752 " \
                  "--save_steps 4750 --seed 42 " \
                  "--data_cache_dir ./output/cache/ " \
                  "--warmup_pct 0.1 --evaluate_during_training " \
                  "--use_devices={} --metric_name={} " \
                  "> {}_{}.out &".format(
                    eval_dataset, model_folder, gpu_idx, eval_dataset,
                    exper_name, run_folder)
        # Set the GPU to use
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        print("Executing:", command)
        os.system(
            'export PYTHONPATH=$PYTHONPATH:$(pwd)/winogrande ; '
            + command)
        folder_number += 1
