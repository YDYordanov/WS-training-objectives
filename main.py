import os
import argparse
import json
import math

import torch
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, \
    get_constant_schedule_with_warmup, AdamW
from tensorboardX import SummaryWriter

from transformer_utils import WGLoader
from bcm_model import BCMModel


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='bert-base-uncased',
                        help="pre-trained model name")
    parser.add_argument("--train_objective", choices=['BWP', 'CSS', 'MAS'],
                        help="training objective name name")
    parser.add_argument('--lowercase', '-lower', action='store_true',
                        help='Do tokenizer lowercasing')
    parser.add_argument("--use_devices", '-gpu', type=str, default=None,
                        help="comma-separated list of devices to run on")
    parser.add_argument('--save_dir', type=str, default='test_run',
                        help="Directory to save the models (for resumption)")
    parser.add_argument('--data_dir', type=str,
                        default='Data/WinoGrande/train_xl.csv',
                        help="Directory to the training data")
    parser.add_argument('--exper_name', type=str, default=None,
                        help="The name of the experiment which "
                             "contains this run")
    parser.add_argument('--log_interval', type=int, default=10e100,
                        help='Evaluate the model every log_interval '
                             'training batches')
    parser.add_argument('--log_weights', '-log_w', action='store_true',
                        help='Log mean and std of the model weights')
    parser.add_argument('--save_interval', '-save', type=int, default=10e100,
                        help='Save the model every log_interval '
                             'training batches')
    parser.add_argument('--evaluate_model', '-eval', type=str, default=None,
                        help='Run evaluation on pre-trained model,'
                             'located in the corresponding folder,'
                             'best_model.pth')
    parser.add_argument('--load_encoder_model', '-load_enc', type=str,
                        default=None,
                        help='Load and evaluate the encoder,'
                             'located on this path')
    parser.add_argument('--pre_evaluate', '-pre_eval', action='store_true',
                        help='Run baseline evaluation before training')
    parser.add_argument('--css_eval', '-css_eval', action='store_true',
                        help='Evaluate on CSS/MAS-specific formatted WSC')
    parser.add_argument('--wsc_evaluate', '-wsc_eval', action='store_true',
                        help='Evaluate on WSC')
    parser.add_argument('--dpr_evaluate', '-dpr_eval', action='store_true',
                        help='Evaluate on DPR')
    parser.add_argument('--wg_evaluate', '-wg_eval', action='store_true',
                        help='Evaluate on WG-dev')
    parser.add_argument('--resume', '-re', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Number of updates steps to accumulate "
                             "before performing a backward/update pass."
                             "This effectively allows for larger batch "
                             "sizes to be used.")

    parser.add_argument('--input_type', '-input',
                        choices=['BERT', 'BERT-basic'],
                        default='BERT-basic',
                        help='The input formatting to Transformer,'
                             'e.g. </s> token formatting')
    parser.add_argument('--use_huggingface_head', '-repo_head',
                        action='store_true',
                        help='Use pre-trained classifier weights')
    parser.add_argument('--use_mlp_head', '-mlp_head',
                        action='store_true',
                        help='use MLP head on top of the attn sums from'
                             'MAS')
    parser.add_argument('--sem_sim_type', '-sem_sim',
                        choices=['cosine', 'additive'], default='cosine',
                        help='Semantic similarity product type;'
                             'see table of semantic similarities')
    parser.add_argument('--context_len', type=int, default=128,
                        help='Cut input sentences to that length')
    parser.add_argument('--num_epochs', '-ep', type=int, default=1,
                        help='Number of training epochs')
    parser.add_argument('--b_size', '-bs', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--dev_b_size', '-dbs', type=int, default=20,
                        help='Evaluation batch size')
    parser.add_argument('--lr', '-lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--scheduler', choices=[None, 'linear', 'constant'],
                        default=None, help='lr scheduler with warm-up')
    parser.add_argument('--warmup_proportion', '-warm_prop', type=float,
                        default=0.1, help='proportion of training data'
                                          ' to warm-up the scheduler')
    parser.add_argument('--model_seed', '-seed', type=int, default=2809,
                        help='Evaluate the model every log_interval '
                             'training batches')

    parser.add_argument('--bert_pooling', '-bert_pool', type=str,
                        default='max',
                        help='Type of pooling to obtain sent emb:'
                             'cls/max/mean')
    parser.add_argument('--pooling', '-pool', choices=['mean','max'],
                        default='mean', help='Type of pooling across tokens')

    args = parser.parse_args()

    if args.use_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.use_devices
        print("Using gpus:", args.use_devices)
    device = torch.device("cuda")

    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed(args.model_seed)
    torch.backends.cudnn.deterministic = True

    vocab_dir = 'vocab_dir'
    data_dir = args.data_dir
    
    log_dir = os.path.join('saved_models', args.save_dir)
    tb_writer = SummaryWriter(logdir=log_dir)

    if args.evaluate_model is None:
        config_file = os.path.join(log_dir, 'config.txt')
        with open(config_file, 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        print(args.__dict__)

    if args.grad_accum_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps "
                         "parameter: {}, should be >= 1".format(
                          args.grad_accum_steps
                          ))
    args.b_size = args.b_size // args.grad_accum_steps

    if 'uncased' in args.model_name:
        do_lower_case = True
        print('Doing lower-casing')
    else:
        do_lower_case = False

    if args.lowercase:
        do_lower_case = True

    tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, do_lower_case=do_lower_case,
            cache_dir=vocab_dir, use_fast=False)
    tokenizer.add_tokens(['[name]', '[number]'])

    print('Vocab size:', tokenizer.vocab_size)

    print('Loading data...')
    if args.css_eval:
        wsc_data_path = 'Data/WSC/WSC_273_winogrande_css_mas.csv'
    else:
        wsc_data_path = 'Data/WSC/WSC_273_winogrande.csv'
    dpr_data_path = 'Data/WSC/DPR(WSCR)/test.csv'

    train_data_path = data_dir
    dev_data_path = 'Data/WinoGrande/dev.csv'
    test_data_path = 'Data/WinoGrande/dev.csv'

    train_loader = WGLoader(
        tokenizer, args.context_len, args.b_size, train_data_path,
        do_train=True).data_loader
    valid_loader = WGLoader(
        tokenizer, args.context_len, args.dev_b_size, dev_data_path,
        do_train=False).data_loader
    test_loader = WGLoader(
        tokenizer, args.context_len, args.dev_b_size, test_data_path,
        do_train=False).data_loader

    eval_loaders = {}

    if args.wsc_evaluate:
        wsc_loader = WGLoader(
            tokenizer, args.context_len, args.dev_b_size, wsc_data_path,
            do_train=False).data_loader
        eval_loaders['WSC'] = wsc_loader

    if args.dpr_evaluate:
        dpr_loader = WGLoader(
            tokenizer, args.context_len, args.dev_b_size, dpr_data_path,
            do_train=False).data_loader
        eval_loaders['DPR'] = dpr_loader

    print('... data loaded!')

    config = {
        'cache_dir': 'pre_trained_models',
        'model_name': args.model_name,
        'emb_dim': 768,
        'd_model': 768
    }

    if 'large' in args.model_name:
        config['d_model'] = 1024
    config['model_name'] = args.model_name
    config['train_objective'] = args.train_objective
    config['pooling'] = args.pooling
    config['bert_pooling'] = args.bert_pooling
    config['vocab_size'] = len(tokenizer)
    config['grad_accum_steps'] = args.grad_accum_steps
    config['use_huggingface_head'] = args.use_huggingface_head
    config['use_attentions'] = True
    config['use_mlp_head'] = args.use_mlp_head
    config['sem_sim_type'] = args.sem_sim_type
    config['log_weights'] = args.log_weights
    print(config)

    model = BCMModel(config)

    print('Loading model to', device)
    model = model.to(device)
    model.device = device
    model = nn.DataParallel(model)
    print('Model loaded!')

    model.module.input_type = args.input_type
    model.module.context_length = args.context_len
    model.module.tokenizer = tokenizer

    model.module.eval_loaders = eval_loaders

    optimizer = AdamW(model.parameters(), lr=args.lr)
    model.module.optimizer = optimizer
    num_training_steps = len(train_loader) * args.num_epochs / \
        args.grad_accum_steps
    if args.scheduler is None:
        model.module.scheduler = None
    elif args.scheduler == 'linear':
        model.module.scheduler = get_linear_schedule_with_warmup(
            optimizer, num_training_steps=num_training_steps,
            num_warmup_steps=math.floor(num_training_steps *
                                        args.warmup_proportion))
    elif args.scheduler == 'constant':
        model.module.scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=math.floor(num_training_steps *
                                        args.warmup_proportion))
    else:
        raise NotImplementedError

    model.module.criterion = nn.CrossEntropyLoss()

    epoch = 1
    train_start_batch = 1
    best_loss = 1.0e+10
    if args.resume:
        print('Recovering from checkpoint...')
        checkpoint = os.path.join(log_dir, 'checkpoint.pth')
        save_dict = torch.load(checkpoint)
        epoch = save_dict['epoch']
        train_start_batch = save_dict['mini_batch'] + 1
        model.module.load_state_dict(save_dict['model_state_dict'])
        model.module.optimizer.load_state_dict(
            save_dict['optimizer_state_dict'])
        if model.module.scheduler is not None:
            model.module.scheduler.load_state_dict(
                save_dict['scheduler_state_dict'])
        print('Model resumed from epoch {} and batch {}.'
              .format(epoch, train_start_batch))

        if train_start_batch >= len(train_loader) + 1:
            epoch += 1

    if args.evaluate_model is not None:
        model_folder = args.evaluate_model
        model_dir = os.path.join(model_folder, 'final_model.pth')
        state_dict = torch.load(model_dir, map_location=device)
        model.module.load_state_dict(state_dict)

        model.eval()

        # Evaluate and save results to file
        if args.wsc_evaluate:
            print('Evaluating on WSC...')
            results = model.module.evaluate(wsc_loader)
            if len(results) > 1:
                accuracy = results[0]
            else:
                accuracy = results
            # save to file (json)
            results_dict = {'wsc_acc': accuracy}
            print('Results saved in:', model_dir)
            with open('{}/WSC.json'.format(model_folder), 'w')\
                    as fp:
                json.dump(results_dict, fp)
        if args.dpr_evaluate:
            print('Evaluating on WSC...')
            results = model.module.evaluate(dpr_loader)
            if len(results) > 1:
                accuracy = results[0]
            else:
                accuracy = results
            # save to file (json)
            results_dict = {'dpr_acc': accuracy}
            print('Results saved in:', model_dir)
            with open('{}/DPR.json'.format(model_folder), 'w') \
                    as fp:
                json.dump(results_dict, fp)
        if args.wg_evaluate:
            print('Evaluating on test set...')
            results = model.module.evaluate(test_loader)
            if len(results) > 1:
                accuracy = results[0]
            else:
                accuracy = results
            # save to file (json)
            results_dict = {'test_acc': accuracy}
            print('Results saved in:', model_dir)
            with open('{}/WG_dev.json'.format(model_folder), 'w') \
                    as fp:
                json.dump(results_dict, fp)

    if args.load_encoder_model is not None:
        model_dir = os.path.join('saved_models', args.load_encoder_model)
        state_dict = torch.load(model_dir, map_location=device)
        # Note: we load the parameters in non-strict fashion
        model.module.encoder.load_state_dict(state_dict, strict=False)

        model.eval()
        model.module.evaluate(valid_loader)

    if args.pre_evaluate:
        model.eval()
        model.module.evaluate(valid_loader)

        for dev_task_name in eval_loaders.keys():
            print(dev_task_name + '...')
            model.module.evaluate(eval_loaders[dev_task_name])

        model.train()

    while epoch <= args.num_epochs:
        print('------')
        print('Epoch', epoch)
        print('------')
        model.train()
        best_loss = model.module.run_epoch(
            epoch, train_loader, valid_loader, tb_writer,
            log_dir, log_interval=args.log_interval,
            save_interval=args.save_interval, start_batch=train_start_batch,
            best_loss=best_loss)

        epoch += 1
        train_start_batch = 1


if __name__ == "__main__":
    main()
