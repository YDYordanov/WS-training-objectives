# WS-training-objectives

This is the code for the paper 
"Does the Objective Matter? 
Comparing Training Objectives for Pronoun Resolution", accepted at EMNLP 2020. 

The code contains implementations of the BWP, MAS and CSS models, 
and the WSC dataset with modified candidates.

The full instructions (with scripts) on how to reproduce the results in the paper 
will be published here by the end of October 2020.

##### Requirements: 

transformers >= 2.5.1,
tensorboardX

##### Example usage:

python main.py --data_dir=Data/WinoGrande/train_xl.csv 
--save_dir=CSS/1 --log_interval=800 
--train_objective=CSS --input_type=BERT-basic 
--scheduler=linear --lr=1e-5 --num_epochs=1 
--model_name=roberta-large --pooling=mean 
--b_size=8 --sem_sim_type=additive 
--exper_name=MyExperiment -seed=1001
