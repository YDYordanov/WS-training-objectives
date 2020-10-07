# WS-training-objectives

This is the code for the paper 
[Does the Objective Matter? 
Comparing Training Objectives for Pronoun Resolution](https://arxiv.org/abs/2010.02570), accepted at EMNLP 2020.

The code contains implementations of the BWP, MAS and CSS models, 
as well as the WSC dataset with modified candidates.

The full instructions (with scripts) on how to reproduce the results in the paper 
will be published here by the end of October 2020.

##### Requirements: 

transformers >= 2.5.1,
tensorboardX

##### Example usage:
```
python main.py \
    --data_dir=Data/WinoGrande/train_xl.csv \
    --save_dir=CSS/1 \
    --log_interval=800 \ 
    --train_objective=CSS \
    --input_type=BERT-basic \ 
    --scheduler=linear \
    --lr=1e-5 \
    --num_epochs=1 \
    --model_name=roberta-large \
    --pooling=mean \ 
    --b_size=8 \
    --sem_sim_type=additive \
    --exper_name=MyExperiment \
    --model_seed=1001
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