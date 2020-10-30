"""
These are utilities for checking for free GPUs
"""
import os
import csv
import time

import torch

from gpuinfo import GPUInfo


def get_pids():
    # Gets all PIDs on all GPUs as a dictionary
    # Each key is a GPU ID
    info = GPUInfo.get_info()

    pids = info[0]
    pids = {value[0]: key for key, value in pids.items()}
    return pids


def gpu_is_empty(gpu_id):
    gpu_id = str(gpu_id)
        
    num_true = 0
    
    if gpu_id in get_pids().keys():
        num_true += 1
        
    time.sleep(15)
    
    if gpu_id in get_pids().keys():
        num_true += 1
    
    if num_true == 0:
        return True
    else:
        return False
 
                
def get_gpu_status(synch_file_path):
    # Get GPU statuses from the GPU sych file
    gpu_synch_file = synch_file_path

    # Read the status
    status = []
    with open(gpu_synch_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            status.append(row)
    return status
    

def find_unassigned_gpu(synch_file_path):
    # Find which gpu is unassigned
    # by looking at the central GPU registry
    
    # It needs to pass two checks one second apart
    print('Looking for available GPU...')
    while True:
        free_gpu_id = -1
        free_gpu_id_2 = -2
        
        status = get_gpu_status(synch_file_path)
        for idx, s in enumerate(status):
            if s == ['0']:
                free_gpu_id = idx
                break
                
        time.sleep(1)
        status = get_gpu_status(synch_file_path)
        for idx, s in enumerate(status):
            if s == ['0']:
                free_gpu_id_2 = idx
                break
                
        if free_gpu_id == free_gpu_id_2:
            print('GPU {} is free!'.format(free_gpu_id))
            return free_gpu_id
        
        time.sleep(1)
        

def update_gpu_synch_file(synch_file_path, gpu_id, is_running=False):
    gpu_synch_file = synch_file_path

    # Read the status
    status = []
    with open(gpu_synch_file, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            status.append(row)

    # Update the status
    status[gpu_id] = [1 if is_running else 0]

    # Write the status
    with open(gpu_synch_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ')
        for value in status:
            writer.writerow(value)
            

def prep_gpus(synch_file_path, taken_gpus_list, num_gpus):
    print('Preparing GPUs...')
    gpu_availability = [str(1) if i in taken_gpus_list else str(0) for i in range(num_gpus)]
    with open(synch_file_path,'w') as f:
        for item in gpu_availability:
            f.write("%s\n" % item)

