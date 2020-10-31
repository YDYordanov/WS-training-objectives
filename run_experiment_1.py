"""
Experiment 1: hyperparameter search for BWP, CSS, MAS.
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

synch_file_path = 'gpu_synch.txt'
# Create an empty gpu_synch file for GPU synchronization
prep_gpus(synch_file_path, taken_gpus_list=args.ignore_gpu_ids,
          num_gpus=args.total_num_gpus)

per_gpu_batch_size = 8

""" BWP Hyperparameters """

train_objective = 'BWP'
exper_name = '{}_hyperparams'.format(train_objective)
folder_number = 1

for b_size in [8, 16]:
    for ep in [3, 4, 5, 8]:
        for lr in [5e-6, 1e-5, 3e-5, 5e-5]:
            for seed in [483957, 328047, 9143]:
                grad_accum_steps = int(b_size) // per_gpu_batch_size

                # Wait for a GPU to free-up and find its idx
                gpu_idx = find_unassigned_gpu(synch_file_path)
                update_gpu_synch_file(synch_file_path, gpu_idx,
                                      is_running=True)

                # Run command
                command = "nohup python main.py " \
                          "--data_path=Data/WinoGrande/train_xl.csv " \
                          "--save_dir=saved_models/{}/{} " \
                          "--input_type=BERT-basic " \
                          "--scheduler=linear -warm_prop=0.1 " \
                          "--log_interval=1000 -bs={} " \
                          "--grad_accum_steps={} " \
                          "-dbs=20 -lr={} --num_epochs={} " \
                          "--model_name=roberta-large -pool=mean " \
                          "--train_objective={} --exper_name={} -seed={} " \
                          "--use_devices={} --gpu_synch_file={} > " \
                          "logs/{}_{}.out &".format(
                            exper_name, folder_number, b_size,
                            grad_accum_steps, lr, ep,
                            train_objective, exper_name, seed, gpu_idx,
                            synch_file_path, exper_name, folder_number)
                # Set the GPU to use
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print("Executing:", command)
                os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
                folder_number += 1

""" CSS Hyperparameters """

train_objective = 'CSS'
exper_name = '{}_hyperparams'.format(train_objective)
folder_number = 1

for b_size in [8, 16]:
    for ep in [3, 4, 5, 8]:
        for lr in [5e-6, 1e-5, 3e-5, 5e-5]:
            for seed in [483957, 328047, 9143]:
                grad_accum_steps = int(b_size) // per_gpu_batch_size

                # Wait for a GPU to free-up and find its idx
                gpu_idx = find_unassigned_gpu(synch_file_path)
                update_gpu_synch_file(synch_file_path, gpu_idx,
                                      is_running=True)

                # Run command
                command = "nohup python main.py " \
                          "--data_path=Data/WinoGrande/train_xl.csv " \
                          "--save_dir=saved_models/{}/{} " \
                          "--input_type=BERT-basic --scheduler=linear " \
                          "-warm_prop=0.1 --log_interval=1000 -bs={} " \
                          "--grad_accum_steps={} " \
                          "-dbs=20 -lr={} --num_epochs={} " \
                          "--model_name=roberta-large -pool=mean " \
                          "-sem_sim=additive --train_objective={} " \
                          "--exper_name={} -seed={} --use_devices={} " \
                          "--gpu_synch_file={} > logs/{}_{}.out &".format(
                            exper_name, folder_number, b_size,
                            grad_accum_steps, lr, ep,
                            train_objective, exper_name, seed, gpu_idx,
                            synch_file_path, exper_name, folder_number)
                # Set the GPU to use
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print("Executing:", command)
                os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
                folder_number += 1

""" MAS Hyperparameters """

train_objective = 'MAS'
exper_name = '{}_hyperparams'.format(train_objective)
folder_number = 1

for b_size in [8, 16]:
    for ep in [3, 4, 5, 8]:
        for lr in [5e-6, 1e-5, 3e-5, 5e-5]:
            for seed in [483957, 328047, 9143]:
                grad_accum_steps = int(b_size) // per_gpu_batch_size

                # Wait for a GPU to free-up and find its idx
                gpu_idx = find_unassigned_gpu(synch_file_path)
                update_gpu_synch_file(synch_file_path, gpu_idx,
                                      is_running=True)

                # Run command
                command = "nohup python main.py " \
                          "--data_path=Data/WinoGrande/train_xl.csv " \
                          "--save_dir=saved_models/{}/{} " \
                          "--input_type=BERT-basic " \
                          "--scheduler=linear -warm_prop=0.1 " \
                          "--log_interval=1000 -bs={} -dbs=20 " \
                          "--grad_accum_steps={} -lr={} " \
                          "--num_epochs={} --model_name=roberta-large " \
                          "-pool=mean --use_mlp_head -sem_sim=additive " \
                          "--train_objective={} --exper_name={} -seed={} " \
                          "--use_devices={} --gpu_synch_file={}" \
                          " > logs/{}_{}.out &".format(
                            exper_name, folder_number, b_size,
                            grad_accum_steps, lr, ep,
                            train_objective, exper_name, seed, gpu_idx,
                            synch_file_path, exper_name, folder_number)
                # Set the GPU to use
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                print("Executing:", command)
                os.system('export PYTHONPATH=$PYTHONPATH:$(pwd) ; ' + command)
                folder_number += 1
                
""" WG-SR Hyperparameters """

exper_name = 'WG-SR_hyperparams'
folder_number = 1
per_gpu_train_batch_size = 8

synch_file_path = 'winogrande/gpu_synch.csv'
# Create an empty gpu_synch file for GPU synchronization
prep_gpus(synch_file_path, taken_gpus_list=args.ignore_gpu_ids,
          num_gpus=args.total_num_gpus)

for lr in [1e-5, 3e-5, 5e-5, 5e-6]:
    for gradient_accumulation_steps in [1, 2]:
        for ep in [3, 4, 5, 8]:
            for seed in [483957, 328047, 9143]:
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
