"""
Experiment 2: Training BWP, CSS and MAS on 20 seeds each
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

synch_file_path = 'gpu_synch2.txt'
# Create an empty gpu_synch file for GPU synchronization
prep_gpus(synch_file_path, taken_gpus_list=args.ignore_gpu_ids,
          num_gpus=args.total_num_gpus)

""" BWP Seeds """

train_objective = 'BWP'
exper_name = '{}_seeds'.format(train_objective)
folder_number = 1

for b_size in [16]:
    for ep in [8]:
        for lr in [1e-5]:
            for seed in [
                43289, 5689, 89304, 868843, 87, 43226, 965, 109, 89324, 679,
                26499, 12924, 87013, 77215, 94508, 44228, 78142, 13704, 57428,
                73493
            ]:
                grad_accum_steps = int(b_size) // 8

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
                          "-wsc_eval -dpr_eval -wg_eval " \
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

""" CSS Seeds """

train_objective = 'CSS'
exper_name = '{}_seeds'.format(train_objective)
folder_number = 1

for b_size in [16]:
    for ep in [8]:
        for lr in [1e-5]:
            for seed in [
                43289, 5689, 89304, 868843, 87, 43226, 965, 109, 89324, 679,
                26499, 12924, 87013, 77215, 94508, 44228, 78142, 13704, 57428,
                73493
            ]:
                grad_accum_steps = int(b_size) // 8

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
                          "-wsc_eval --css_eval -dpr_eval -wg_eval " \
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

""" MAS Seeds """

train_objective = 'MAS'
exper_name = '{}_seeds'.format(train_objective)
folder_number = 1

for b_size in [8]:
    for ep in [8]:
        for lr in [1e-5]:
            for seed in [
                43289, 5689, 89304, 868843, 87, 43226, 965, 109, 89324, 679,
                26499, 12924, 87013, 77215, 94508, 44228, 78142, 13704, 57428,
                73493
            ]:
                grad_accum_steps = int(b_size) // 8

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
                          "-wsc_eval --css_eval -dpr_eval -wg_eval " \
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
