# WS-training-objectives

This repository contains the code to reproduce
 the results of the paper: 
[Does the Objective Matter? 
Comparing Training Objectives for Pronoun Resolution](https://arxiv.org/abs/2010.02570), accepted at EMNLP 2020.

This repository contains implementations of the BWP, MAS and CSS training objectives,
as well as the code for the WG-SR training objective, adapted 
from the [WinoGrande code repository](https://github.com/allenai/winogrande). 
This repository also contains the WSC dataset with modified candidates. 
The WSC dataset versions can be found in Data/WSC.

Please follow the instructions below to reproduce the results of our paper.

### Requirements

pytorch

transformers >= 2.5.1

pytorch-transformers == 1.1.0

tensorboardX

gpuinfo

scipy

sklearn


## Set-up the Data

The WinoGrande dataset can be found at the following [link](
https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip), or at [
http://winogrande.allenai.org/](http://winogrande.allenai.org/). 
Download and unpack the individual dataset files in Data/WinoGrande.

The WSC dataset (modified and unmodified versions) can be found in Data/WSC.

The DPR (test) dataset can be found on [Ariba Siddiqui's Kaggle webpage for DPR](
https://www.kaggle.com/ariba05/definite-pronoun-resolution-dataset).
Note that the DPR test dataset is named "winograd_train.csv", 
which is the file you should download. We use this version of the dataset, 
since it has exact pronoun location of a single pronoun 
to be disambiguated per dataset entry (14 entries in the original 
DPR test dataset contain multiple mentions of the same pronoun, 
and we only wish to disambiguate one of them). 
Download and unpack the individual dataset files in Data/DPR.
Alternatively, we have provided the processed DPR dataset
in Data/DPR/DPR_test.jsonl.

After following all of the instructions above, execute in Data/DPR:
```
python extract_data.py
```

Then execute in the main directory:

```
python prepare_all_data.py
```


## Reproducing Experiment 1: Hyperparameter Search

Run run_experiment_1.py and then 
run_experiment_1_WG-SR.py 
to train all 4 models 
over the hyperparameter space.

Example usage:
```
python run_experiment_1.py \
    --total_num_gpus=8 \
    --ignore_gpu_ids 0 1 2 3

python run_experiment_1_WG-SR.py \
    --total_num_gpus=8 \
    --ignore_gpu_ids 4 5 6 7
```

After training the models, 
print the aggregated results and the best 
hyperparameters by running each of the commands below. 

```
python collect_results.py --dirs saved_models/BWP_hyperparams \
    --best_hyperparams --result_files final_result.json
python collect_results.py --dirs saved_models/CSS_hyperparams \
    --best_hyperparams --result_files final_result.json
python collect_results.py --dirs saved_models/MAS_hyperparams \
    --best_hyperparams --result_files final_result.json
python collect_results.py --dirs winogrande/saved_models/WG-SR_hyperparams \
    --best_hyperparams --result_files metrics.json
```

If prompted to choose a metric to report, 
choose the corresponding accuracy.

## Reproducing Experiment 2: Multi-Seed Model Performance on WG, WSC and DPR

Run run_experiment_2.py and then 
run_experiment_2_WG-SR.py to train on 20 seeds all 4 models 
with the best hyperparameters from Experiment 1, and to evaluate them on 
WG-dev, WSC and DPR. Note that the best hyperparameters are pre-set in 
run_experiment_2.py, so you can run it independently.

The script also trains WG-SR on additional 10 seeds.

Example usage:
```
python run_experiment_2.py \
    --total_num_gpus=8 \
    --ignore_gpu_ids 0 1 2 3

python run_experiment_2_WG-SR.py \
    --total_num_gpus=8 \
    --ignore_gpu_ids 4 5 6 7
```

After training the models, collect the results by running:
```
python collect_results.py --dirs saved_models/BWP_seeds \
    --result_files WSC.json DPR.json WG-dev.json
python collect_results.py --dirs saved_models/CSS_seeds \
    --result_files WSC.json DPR.json WG-dev.json
python collect_results.py --dirs saved_models/MAS_seeds \
    --result_files WSC.json DPR.json WG-dev.json
python collect_results.py --dirs winogrande/saved_models/WG-SR_seeds \
    --result_files WSC.json DPR.json WG-dev.json
```


## Reference
```
@inproceedings{EMNLP-2020-2271,
  title = "Does the Objective Matter? Comparing Training Objectives for Pronoun Resolution",
  author = "Yordan Yordanov and Oana-Maria Camburu and Vid Kocijan and Thomas Lukasiewicz",
  year = "2020",
  booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, EMNLP 2020, November 16--20, 2020",
  month = "November",
}
```
