import json
from tqdm import tqdm

import torch
import torch.nn as nn

from transformer_utils import log_weights
from model import TransformerModel


class BCMModel(TransformerModel):
    """
    These are the BWP, CSS and MAS models
    """
    def __init__(self, config):
        super(BCMModel, self).__init__(config)

        self.train_objective = config['train_objective']

        # Dictionary of additional data loaders
        # for evaluation on DPR and WSC
        self.eval_loaders = None

        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.softmax = nn.Softmax(dim=1)
        self.log_sigmoid = nn.LogSigmoid()

        if self.config['sem_sim_type'] == 'cosine':
            self.cosine_sim = nn.CosineSimilarity(dim=1)
        elif self.config['sem_sim_type'] == 'additive':
            self.sem_sim = nn.Sequential(
                nn.Linear(2 * self.d_model, self.d_model),
                nn.Tanh(),
                nn.Linear(self.d_model, 1)
            )
        else:
            raise NotImplementedError

        if self.config['use_mlp_head']:
            # Add an MLP on top of the two candidate attention sums for MAS
            # num_attns is how many attn heads there are in all layers
            num_attns = self.encoder.config.num_attention_heads * \
                self.encoder.config.num_hidden_layers
            self.mlp_head = nn.Sequential(
                nn.Linear(2 * num_attns, 2 * num_attns),
                nn.Tanh(),
                nn.Linear(2 * num_attns, 2 * num_attns),
                nn.Tanh(),
                nn.Linear(2 * num_attns, 2)
            )

    def mas_forward(self, attentions, cand_select_mask0, cand_select_mask1,
                    cand_select_mask2):

        # Collect the attentions from all layers
        attentions = [attn.unsqueeze(dim=4) for attn in attentions]
        attentions = torch.cat(attentions, dim=4).permute(0, 4, 1, 2, 3)
        # This is a tensor of shape:
        # (b_size, num_heads, num_layers, seq_len, seq_len)

        num_selected0 = cand_select_mask0.sum(dim=1)
        num_selected1 = cand_select_mask1.sum(dim=1)
        num_selected2 = cand_select_mask2.sum(dim=1)
        for b_id in num_selected0.long().tolist():
            assert b_id == 1
        for b_id in num_selected1.long().tolist():
            assert b_id >= 1
        for b_id in num_selected2.long().tolist():
            assert b_id >= 1

        # Now select the attentions of the <mask> token to all other tokens
        # First we need to turn the mask token selection mask
        # from (b_size, seq_len) to the format above:
        mask_token_select_mask = cand_select_mask0.view(
            cand_select_mask0.size(0), 1, 1, cand_select_mask0.size(1), 1)
        mask_token_select_mask = mask_token_select_mask.repeat(
            1, attentions.size(1), attentions.size(2), 1,
            attentions.size(4))
        mask_attends_to = (attentions * mask_token_select_mask).sum(dim=3)

        # We aim to get a MAS-like score for each candidate

        # First apply the candidate masks to the attentions,
        # and average across the tokens of a candidate
        # print(selection_mask1.size(), attentions.size())
        cand1_select_mask = cand_select_mask1.view(
            cand_select_mask1.size(0), 1, 1, cand_select_mask1.size(1))
        cand1_select_mask = cand1_select_mask.repeat(
            1, attentions.size(1), attentions.size(2), 1)
        cand1_attentions = (mask_attends_to * cand1_select_mask
                            ).sum(dim=-1) / (
            num_selected1.view(num_selected1.size(0), 1, 1).repeat(
                1, attentions.size(1), attentions.size(2)))
        cand2_select_mask = cand_select_mask2.view(
            cand_select_mask2.size(0), 1, 1, cand_select_mask2.size(1))
        cand2_select_mask = cand2_select_mask.repeat(
            1, attentions.size(1), attentions.size(2), 1)
        cand2_attentions = (mask_attends_to * cand2_select_mask
                            ).sum(dim=-1) / (
            num_selected2.view(num_selected2.size(0), 1, 1).repeat(
                1, attentions.size(1), attentions.size(2)))

        # Then compute the max-masks of MAS
        candidate2_mask = torch.argmax(torch.cat(
            (cand1_attentions.unsqueeze(dim=3),
             cand2_attentions.unsqueeze(dim=3)), dim=3), dim=3)
        candidate1_mask = - candidate2_mask + 1
        # Apply the max-masks
        cand1_attentions *= candidate1_mask
        cand2_attentions *= candidate2_mask

        b_size = cand1_attentions.size(0)
        # print(cand1_attentions.size())

        if self.config['use_mlp_head']:
            both_cand_attentions = torch.cat(
                (cand1_attentions.view(b_size, -1),
                 cand2_attentions.view(b_size, -1)), dim=1)
            similarities = self.mlp_head(both_cand_attentions)
        else:
            # Sum across the heads and layers
            cand1_total_attn = cand1_attentions.sum(dim=1).sum(dim=1)
            cand2_total_attn = cand2_attentions.sum(dim=1).sum(dim=1)
            similarities = torch.cat((
                cand1_total_attn.view(-1, 1), cand2_total_attn.view(-1, 1)),
                dim=1)
        return similarities

    def forward(self, sentence, lik_select_mask1, lik_select_mask2,
                cand_select_mask0, cand_select_mask1, cand_select_mask2,
                token_type_ids=None, attention_mask=None):
        """
        Inputs:
        - sentence of shape [b_size, seq_len],
        - selection_mask1,2 for the word emb-s of
            both candidates
        - selection_mask0 for word emb of <mask> token
        - attention_mask is optional tensor of shape
        [b_size, seq_len, seq_len]
        - token_type_ids, also known as 'segment_ids'

        Outputs:
        - similarity scores for CSS
        - prediction scores for BWP
        - log-probability of the first candidate
        - log-probability of the second candidate
        - similarity scores for MAS (Maximum Attention Score)
        """

        self.encoder.config.output_hidden_states = True
        prediction_scores, hidden_states, attentions = self.encoder(
            sentence,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        top_hidden_states = hidden_states[-1]

        # First let's compute the output of the CorefSemSim model:
        mask_word_vectors = top_hidden_states * cand_select_mask0.unsqueeze(
            dim=2).repeat(1, 1, top_hidden_states.size(2))
        candidate1_vectors = top_hidden_states * cand_select_mask1.unsqueeze(
            dim=2).repeat(1, 1, top_hidden_states.size(2))
        candidate2_vectors = top_hidden_states * cand_select_mask2.unsqueeze(
            dim=2).repeat(1, 1, top_hidden_states.size(2))

        # Pool
        if self.config['pooling'] == 'mean':
            num_selected0 = cand_select_mask0.sum(dim=1)
            num_selected1 = cand_select_mask1.sum(dim=1)
            num_selected2 = cand_select_mask2.sum(dim=1)
            for b_id in num_selected0.long().tolist():
                assert b_id >= 1
            for b_id in num_selected1.long().tolist():
                assert b_id >= 1
            for b_id in num_selected2.long().tolist():
                assert b_id >= 1
            mask_word_emb = mask_word_vectors.sum(dim=1) \
                / num_selected0.unsqueeze(dim=1)
            candidate1_emb = candidate1_vectors.sum(dim=1) \
                / num_selected1.unsqueeze(dim=1)
            candidate2_emb = candidate2_vectors.sum(dim=1) \
                / num_selected2.unsqueeze(dim=1)
        else:
            raise NotImplementedError

        # Take semantic similarity product:
        if self.config['sem_sim_type'] == 'cosine':
            similarity1 = self.cosine_sim(mask_word_emb, candidate1_emb)
            similarity2 = self.cosine_sim(mask_word_emb, candidate2_emb)
        elif 'additive' in self.config['sem_sim_type']:
            similarity1 = self.sem_sim(
                torch.cat((mask_word_emb, candidate1_emb), dim=1))
            similarity2 = self.sem_sim(
                torch.cat((mask_word_emb, candidate2_emb), dim=1))
        else:
            raise NotImplementedError

        similarities = torch.cat((
            similarity1.view(-1, 1), similarity2.view(-1, 1)),
            dim=1)

        # Now for the BinaryWordPred model:
        # Take softmax
        # print(prediction_scores.size())
        prediction_probs = self.logsoftmax(prediction_scores)
        # Apply selection mask for each candidate word:
        selected_probs_1 = prediction_probs * lik_select_mask1
        selected_probs_2 = prediction_probs * lik_select_mask2

        # Pool
        if self.config['pooling'] == 'mean':
            num_selected1 = lik_select_mask1.sum(dim=2).sum(dim=1)
            num_selected2 = lik_select_mask2.sum(dim=2).sum(dim=1)
            log_lik_1 = selected_probs_1.sum(dim=2).sum(dim=1) \
                / (num_selected1 + 1.0e-16)
            log_lik_2 = selected_probs_2.sum(dim=2).sum(dim=1) \
                / (num_selected2 + 1.0e-16)
        else:
            raise NotImplementedError
        for b_id in num_selected1.long().tolist():
            assert b_id >= 1
        for b_id in num_selected2.long().tolist():
            assert b_id >= 1

        mas_similarities = self.mas_forward(
            attentions, cand_select_mask0, cand_select_mask1,
            cand_select_mask2)

        return similarities, prediction_scores, log_lik_1, log_lik_2, \
            mas_similarities

    def evaluate(self, loader):
        num_all = 0
        num_bwp_correct = 0
        num_css_correct = 0
        num_mas_correct = 0
        total_mas_loss = 0
        total_css_loss = 0
        total_bwp_loss = 0
        for step, batch in tqdm(enumerate(loader),
                                total=len(loader)):
            with torch.no_grad():
                input_dict, y = batch
                input_dict = self.send_dict_to_device(input_dict)
                y = y.to(self.device)
                # Convert truthfulness to binary class label
                class_labels = 1 - y

                similarities, prediction_scores, probs1, probs2,\
                    mas_similarities = self.forward(**input_dict)

                bwp_outputs = torch.cat((
                    probs2.view(-1, 1), probs1.view(-1, 1)), dim=1)
                bwp_loss = self.criterion(bwp_outputs, y)
                total_bwp_loss += bwp_loss.item()

                css_loss = self.criterion(similarities, class_labels)
                mas_loss = self.criterion(mas_similarities, class_labels)
                total_css_loss += css_loss.item()
                total_mas_loss += mas_loss.item()

                b_size = prediction_scores.size(0)

                # Make predictions
                bwp_probs = self.softmax(
                    bwp_outputs)[:, torch.LongTensor([1, 0])]
                css_probs = self.softmax(similarities)
                mas_probs = self.softmax(mas_similarities)
                bwp_preds_list = [1 if bwp_probs[i][0] >= bwp_probs[i][1]
                                  else 0 for i in range(b_size)]
                css_preds_list = [1 if css_probs[i][0] >= css_probs[i][1]
                                  else 0 for i in range(b_size)]
                mas_preds_list = [1 if mas_probs[i][0] >= mas_probs[i][1]
                                  else 0 for i in range(b_size)]

                # Get the number of correct predictions
                y_list = y.type(torch.long).tolist()
                bwp_correct = [None for i in range(len(y_list))
                               if y_list[i] == bwp_preds_list[i]]
                num_bwp_correct += len(bwp_correct)
                css_correct = [None for i in range(len(y_list))
                               if y_list[i] == css_preds_list[i]]
                num_css_correct += len(css_correct)
                mas_correct = [None for i in range(len(y_list))
                               if y_list[i] == mas_preds_list[i]]
                num_mas_correct += len(mas_correct)
                num_all += b_size

        av_mas_loss = total_mas_loss / len(loader)
        av_css_loss = total_css_loss / len(loader)
        av_bwp_loss = total_bwp_loss / len(loader)
        mas_acc = num_mas_correct / num_all
        css_acc = num_css_correct / num_all
        bwp_acc = num_bwp_correct / num_all
        print('Acc mas/css/bwp:', round(mas_acc, 3), round(css_acc, 3),
              round(bwp_acc, 3),
              'Average losses mas/css/bwp:', round(av_mas_loss, 4),
              round(av_css_loss, 4), round(av_bwp_loss, 4))
        result_dict = {
            'mas_acc': mas_acc,
            'css_acc': css_acc,
            'bwp_acc': bwp_acc,
            'av_mas_loss': av_mas_loss,
            'av_css_loss': av_css_loss,
            'av_bwp_loss': av_bwp_loss,
        }
        return result_dict

    def run_epoch(self, epoch, train_loader, valid_loader, tb_writer,
                  save_dir, log_interval=200, save_interval=2000,
                  start_batch=1, best_loss=1.0e+10):
        best_acc = 0.5
        accuracy = 0.0
        total_mas_loss = 0
        total_css_loss = 0
        total_bwp_loss = 0
        buffer_mas_loss = 0
        buffer_css_loss = 0
        buffer_bwp_loss = 0

        for step, batch in tqdm(enumerate(train_loader),
                                total=len(train_loader)):
            # Fast-forward the iterator until the resumption point:
            if start_batch > 1 and step + 1 < start_batch:
                continue
            elif 1 < start_batch == (step + 1):
                print('Resuming training...')

            input_dict, y = batch
            input_dict = self.send_dict_to_device(input_dict)
            y = y.to(self.device)
            # Convert truthfulness to binary class label
            class_labels = 1 - y

            # print(self.tokenizer.convert_ids_to_tokens(
            #     input_dict['sentence'][0].tolist()))

            similarities, prediction_scores, probs1, probs2, \
                mas_similarities = self.forward(**input_dict)

            bwp_outputs = torch.cat((
                probs2.view(-1, 1), probs1.view(-1, 1)), dim=1)
            bwp_loss = self.criterion(bwp_outputs, y)
            total_bwp_loss += bwp_loss.item()

            css_loss = self.criterion(similarities, class_labels)
            mas_loss = self.criterion(mas_similarities, class_labels)
            total_css_loss += css_loss.item()
            total_mas_loss += mas_loss.item()

            buffer_css_loss += css_loss.item()
            buffer_mas_loss += mas_loss.item()
            buffer_bwp_loss += bwp_loss.item()

            if self.train_objective == 'BWP':
                loss = bwp_loss
            elif self.train_objective == 'CSS':
                loss = css_loss
            else:
                loss = mas_loss
            loss.backward()

            # For gradient accumulation we accumulate the
            # gradients by not calling optimizer.zero_grad()
            # until this moment.
            if (step + 1) % self.config['grad_accum_steps'] == 0:
                self.optimizer.step()  # Update the parameters
                if self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()  # Stop grad accumulation

            if (step + 1) % log_interval == 0:
                print('Evaluation ...')
                self.eval()
                result_dict = self.evaluate(valid_loader)
                log_step = step + len(train_loader) * (epoch - 1)
                tb_writer.add_scalar(
                    'Accuracy(MAS)', result_dict['mas_acc'], log_step + 1)
                tb_writer.add_scalar(
                    'Accuracy(CSS)', result_dict['css_acc'], log_step + 1)
                tb_writer.add_scalar(
                    'Accuracy(BWP)', result_dict['bwp_acc'], log_step + 1)
                tb_writer.add_scalar(
                    'Dev mas loss', result_dict['av_mas_loss'], log_step + 1)
                tb_writer.add_scalar(
                    'Dev css loss', result_dict['av_css_loss'], log_step + 1)
                tb_writer.add_scalar(
                    'Dev bwp loss', result_dict['av_bwp_loss'], log_step + 1)

                buffer_av_mas_loss = buffer_mas_loss / log_interval
                tb_writer.add_scalar('Train mas loss', buffer_av_mas_loss,
                                     log_step + 1)
                buffer_av_css_loss = buffer_css_loss / log_interval
                tb_writer.add_scalar('Train css loss', buffer_av_css_loss,
                                     log_step + 1)
                buffer_av_bwp_loss = buffer_bwp_loss / log_interval
                tb_writer.add_scalar('Train bwp loss', buffer_av_bwp_loss,
                                     log_step + 1)
                print('Current train losses mas/css/bwp:',
                      round(buffer_av_mas_loss, 3),
                      round(buffer_av_css_loss, 3),
                      round(buffer_av_bwp_loss, 3))

                buffer_mas_loss = 0
                buffer_css_loss = 0
                buffer_bwp_loss = 0

                if self.train_objective == 'BWP':
                    accuracy = result_dict['bwp_acc']
                elif self.train_objective == 'CSS':
                    accuracy = result_dict['css_acc']
                else:
                    accuracy = result_dict['mas_acc']

                print('Additional evaluation...')
                for task in self.eval_loaders.keys():
                    print(task + '...')
                    result_dict = self.evaluate(self.eval_loaders[task])
                    tb_writer.add_scalar(
                        task + 'accuracy(mas)', result_dict['mas_acc'],
                        log_step + 1)
                    tb_writer.add_scalar(
                        task + 'accuracy(css)', result_dict['css_acc'],
                        log_step + 1)
                    tb_writer.add_scalar(
                        task + 'accuracy(bwp)', result_dict['bwp_acc'],
                        log_step + 1)
                if self.log_weights:
                    log_weights(self, log_step, tb_writer)

                self.train()

            if (step + 1) % save_interval == 0:
                print('Saving model...')
                torch.save(self.state_dict(),
                           '{}/model.pth'.format(save_dir))
                self.save(save_dir, epoch, step, accuracy)

                if accuracy > best_acc:
                    torch.save(self.state_dict(),
                               '{}/best_model.pth'.format(save_dir))
                    best_acc = accuracy
                print('... model saved!')

        ep_mas_loss = total_mas_loss / len(train_loader)
        ep_css_loss = total_css_loss / len(train_loader)
        ep_bwp_loss = total_bwp_loss / len(train_loader)
        print('Epoch train losses mas/css/bwp:',
              round(ep_mas_loss, 3), round(ep_css_loss, 3),
              round(ep_bwp_loss, 3))

        print('Saving final model...')
        torch.save(self.state_dict(),
                   '{}/final_model.pth'.format(save_dir))
        print('Evaluating final model...')
        self.eval()
        result_dict = self.evaluate(valid_loader)
        with open('{}/final_result.json'.format(save_dir), 'w') as fp:
            json.dump(result_dict, fp)

        return best_loss
