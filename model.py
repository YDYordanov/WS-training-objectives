import os

import torch
import torch.nn as nn
from transformers import AutoModel, RobertaForSequenceClassification, \
    RobertaForMaskedLM, BertForMaskedLM, AlbertForMaskedLM, AutoConfig


class BERTEncoder(nn.Module):
    def __init__(self, config):
        super(BERTEncoder, self).__init__()
        self.config = config
        if 'bert_pooling' in config.keys():
            self.pooling = config['bert_pooling']
        else:
            self.pooling = 'cls'

        if self.config['use_huggingface_head']:
            if 'roberta' in self.config['model_name']:

                model = RobertaForSequenceClassification.from_pretrained(
                    config['model_name'], cache_dir=config['cache_dir'],
                    output_attentions=config['use_attentions'])
                self.bert = model.roberta
                self.classifier = model.classifier
            else:
                raise NotImplementedError
        else:
            self.bert_config = AutoConfig.from_pretrained(
                config['model_name'])
            self.bert_config.output_attentions = config['use_attentions']
            self.bert = AutoModel.from_pretrained(
                config['model_name'], cache_dir=config['cache_dir'],
                config=self.bert_config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):

        # This is a quick fix for a known RoBERTa issue
        # with segment ids:
        # https://github.com/huggingface/transformers/issues/1234
        if 'roberta' in self.config['model_name']:
            outputs = self.bert(input_ids,
                                attention_mask=attention_mask)
        else:
            outputs = self.bert(input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

        if self.config['use_attentions']:
            return outputs
        else:
            top_hidden_states, bert_pooled_output = outputs

            if self.pooling == 'cls':
                pooled_output = bert_pooled_output
            elif self.pooling == 'mean':
                pooled_output = top_hidden_states.mean(dim=1)
            elif self.pooling == 'max':
                pooled_output = top_hidden_states.max(dim=1)[0]
            else:
                raise NotImplementedError

            return top_hidden_states, pooled_output


class TransformerModel(nn.Module):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.config = config
        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.emb_dim = config['emb_dim']
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.device = None
        self.context_length = None
        self.input_type = None
        self.log_weights = config['log_weights']

        if 'roberta' in self.config['model_name']:
            print('Loading RobertaForMaskedLM...')
            self.encoder = RobertaForMaskedLM.from_pretrained(
                config['model_name'], cache_dir=config['cache_dir'],
                output_hidden_states=True,
                output_attentions=config['use_attentions'])
        else:
            raise NotImplementedError
        print('Model loaded!')

    def save(self, save_dir, epoch, step, dev_loss):
        checkpoint = os.path.join(save_dir, 'checkpoint.pth')
        if self.scheduler is not None:
            scheduler_dict = self.scheduler.state_dict()
        else:
            scheduler_dict = None
        torch.save({
            'epoch': epoch,
            'mini_batch': step + 1,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': scheduler_dict,
            'loss': dev_loss
        }, checkpoint)

    def send_dict_to_device(self, input_dict):
        for key in input_dict.keys():
            input_dict[key] = input_dict[key].to(self.device)
        return input_dict
