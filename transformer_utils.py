"""
These are the utilities used for text processing,
batching and tokenisation.
"""

import os
import sys
import re
import csv
import copy

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader


csv.field_size_limit(sys.maxsize)


class WGDataset(Dataset):
    """
    This class processes the WinoGrande dataset
    for BWP, CSS and MAS.
    """

    def __init__(self, data_path, tokenizer, context_length=512):
        super(WGDataset, self).__init__()

        self.tokenizer = tokenizer
        self.data_path = data_path
        self.context_length = context_length
        self.fast_forward = False

        # Load the data; it should be of the form:
        # (label, original_sent, sent1, sent2)
        data_file = os.path.join(self.data_path)
        with open(data_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            data = list(reader)

        # Reformat the data as a list of sentences,
        # each one being correct or incorrect
        # 0 denotes incorrect and 1 denotes correct
        self.data = []
        for entry in data:
            label = entry[0]
            alternative1 = entry[2]
            alternative2 = entry[3]
            sent = entry[1]
            self.data.append([1-int(label), sent, alternative1, alternative2])
        print('data size:', len(data), 'Expanded:', len(self.data))

    def __len__(self):
        return len(self.data)

    def find_candidate_ids(self, sent, word):
        word = word.lower()
        sent = [w.lower() for w in sent]
        word_stripped = word.replace(" ", "")
        ids_in_text = []
        try:
            ids_in_text = [m.start() for m in re.finditer(
                r'%s' % word_stripped, r'%s' % (''.join(sent)))]
        except:
            print('word:', word_stripped, 'sent:', sent)
        else:
            ids_in_text = [m.start() for m in re.finditer(
                r'%s' % word_stripped, r'%s' % (''.join(sent)))]

        # print(ids_in_text)
        # print(sent, ''.join(sent), word, word_stripped)
        if len(ids_in_text) == 0:
            print('0 candidate ids found')
        # assert len(ids_in_text) >= 1

        word_pos_ids = []
        for start_id in ids_in_text:
            start_string = ''.join(sent)[:start_id]
            for i, toki in enumerate(sent):
                if ''.join(sent[:i]) == start_string:
                    for j, tokj in enumerate(sent[i:]):
                        word_variants = [word, word+'s', word+'es',
                                         word_stripped, word_stripped+'s',
                                         word_stripped+'es']
                        if ''.join(sent[i:i + j]) in word_variants:
                            new_ids = [i for i in range(i, i + j)]

                            word_pos_ids += new_ids

        if word_pos_ids == []:
            for i, toki in enumerate(sent):
                if word in toki:
                    word_pos_ids += [i]

        # word_pos_ids = sorted(set(word_pos_ids))
        # assert word_pos_ids != []
        return word_pos_ids

    def find_candidates(self, sent, candidate):
        # Here we find the position(s) of the tokens of
        # the candidates in the masked sentence;
        # also we return the masked sentence

        base_sent = copy.deepcopy(sent)

        # First we mask and tokenize
        mask_token = self.tokenizer.mask_token
        masked_sent = re.sub('_', mask_token, base_sent)
        masked_sent = self.tokenizer.tokenize(masked_sent)

        # Temporarily remove 'Ġ' from RoBERTa tokens
        # and # from BERT tokens for consistency purposes
        masked_sent_stripped = [re.sub('Ġ', '', tok) for tok in masked_sent]
        masked_sent_stripped = [re.sub('#', '', tok)
                                for tok in masked_sent_stripped]

        # Find pos ids of candidate
        candidate_pos_ids = self.find_candidate_ids(
            masked_sent_stripped, candidate)

        if len(candidate_pos_ids) == 0:
            print(masked_sent_stripped, candidate)

        # Account for special tokens (cls-equivalent)
        candidate_pos_ids = [pos+1 for pos in candidate_pos_ids]

        return candidate_pos_ids, masked_sent

    def find_replacements(self, sent, candidate):
        # Here we find the position(s) of the tokens of
        # the substituted candidate in the substituted
        # sentence, along with its token ids

        # First get both sentences properly tokenized
        base_sent = copy.deepcopy(sent)

        # Lowercase the first letter of the candidate
        # so that it fits in the sentence
        # (whenever it makes sense)
        annoying_words = ['The ', 'His ', 'Her ', 'My ']
        for word in annoying_words:
            if candidate.startswith(word):
                candidate = word.lower() + candidate[len(word):]
        #print('candidate:', candidate)
        filled_sent = re.sub('_', candidate, base_sent)
        filled_sent = self.tokenizer.tokenize(filled_sent)
        base_sent = self.tokenizer.tokenize(base_sent)

        # Temporarily remove 'Ġ' from RoBERTa tokens
        # and # from BERT tokens
        # in both sentences, for consistency purposes
        base_sent = [re.sub('Ġ', '', tok) for tok in base_sent]
        filled_sent_stripped = [re.sub('Ġ', '', tok) for tok in filled_sent]
        base_sent = [re.sub('#', '', tok) for tok in base_sent]
        filled_sent_stripped = [re.sub('#', '', tok)
                                for tok in filled_sent_stripped]

        # Compare both token strings for differences:
        word_start = None
        word_end = None
        for i in range(len(filled_sent)):
            if filled_sent_stripped[i] != base_sent[i]:
                word_start = i
                break
        for i in range(1, len(filled_sent) + 1):
            if filled_sent_stripped[-i] != base_sent[-i]:
                word_end = len(filled_sent) - i
                break
        assert word_start is not None
        assert word_end is not None
        if word_start > word_end:
            print(word_start, word_end)
            print(base_sent)
            print(filled_sent_stripped)
            print(len(base_sent), len(filled_sent_stripped), len(filled_sent))
        assert word_start <= word_end

        replacement_ids = [i for i in range(word_start, word_end + 1)]
        token_ids = [self.tokenizer.convert_tokens_to_ids(filled_sent[i])
                     for i in replacement_ids]

        mask_token = self.tokenizer.mask_token
        masked_sent = filled_sent[:replacement_ids[0]] + [mask_token] \
            + filled_sent[replacement_ids[-1] + 1:]
        return replacement_ids, token_ids, masked_sent

    def __getitem__(self, idx):
        data = self.data[idx]

        # Do deepcopy in order to avoid changing the underlying data
        label = copy.deepcopy(data[0])
        sent = copy.deepcopy(data[1])
        candidate = copy.deepcopy(data[2])
        candidate2 = copy.deepcopy(data[3])  # the other candidate

        """
        We need the token ids of each candidate word,
        for selection masking of the log-likelihood.
        To do this tokenizer-agnostic, we need to 
        substitute each candidate and tokenize the 
        resulting sentence(s).
        """
        _, candidate_ids, masked_sent_v1 = self.find_replacements(
            sent, candidate)
        _, candidate2_ids, _ = self.find_replacements(sent, candidate2)

        """
        Find the pos_ids of both candidates, 
        as to where they occur in the sentence.
        Later I can use the selection masks w.r.t.
        those candidates to compute their embeddings
        in the masked sentence.
        """
        candidate_pos_ids, masked_sent_v2 = self.find_candidates(
            sent, candidate)
        candidate2_pos_ids, _ = self.find_candidates(sent, candidate2)

        # Choosing masked_sent_v1 since it makes more sense
        sent_ids = self.tokenizer.convert_tokens_to_ids(masked_sent_v1)
        segment_ids = self.tokenizer.create_token_type_ids_from_sequences(
            sent_ids
        )

        # Add [cls] and [sep]-equivalent tokens
        sent_ids = self.tokenizer.build_inputs_with_special_tokens(
            sent_ids)
        sent_ids = copy.deepcopy(sent_ids)

        if candidate_pos_ids == []:
            print('Failed to find candidate1!')
            candidate_pos_ids = range(len(sent_ids))
        if candidate2_pos_ids == []:
            print('Failed to find candidate2!')
            candidate2_pos_ids = range(len(sent_ids))

        mask_token = self.tokenizer.mask_token
        masked_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        mask_id = None
        for i, tok in enumerate(sent_ids):
            if tok == masked_token_id:
                mask_id = [i]
        assert mask_id is not None

        return_dict = {
            'sent_ids': sent_ids,
            'candidate_ids': candidate_ids,
            'candidate2_ids': candidate2_ids,
            'candidate_pos_ids': candidate_pos_ids,
            'candidate2_pos_ids': candidate2_pos_ids,
            'mask_id': mask_id,
            'label': label,
            'segment_ids': segment_ids,
        }

        return return_dict


class WGLoader:
    def __init__(self, tokenizer, context_length,
                 batch_size, data_path, do_train=False, drop_last=False):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.do_train = do_train
        self.drop_last = drop_last

        # Initialise dataset
        self.dataset = WGDataset(
            data_path=self.data_path,
            context_length=self.context_length,
            tokenizer=self.tokenizer
        )

        # Define function to process the output
        # of the DataLoader
        def collate_fn(input_list):
            def get_candidate_selection_mask(selected_pos_ids_list):
                # Creating selection mask for token embeddings
                # of a given candidate
                # Note: it should be of size:
                # (b_size, seq_len).
                selection_mask = np.zeros((b_size, seq_len),
                                          dtype=int)

                for i in range(b_size):
                    for j in selected_pos_ids_list[i]:
                        selection_mask[i, j] = 1
                selection_mask = torch.from_numpy(selection_mask).float()
                return selection_mask

            def get_likelihood_selection_mask(selected_ids_list, mask_id_list):
                # Creating selection mask for log-likelihoods
                # Note: it should be of size:
                # (b_size, seq_len, vocab_size).
                selection_mask = np.zeros((b_size, seq_len, vocab_size),
                                          dtype=int)

                for i in range(b_size):
                    j = mask_id_list[i]
                    for k in selected_ids_list[i]:
                        selection_mask[i, j, k] = 1
                selection_mask = torch.from_numpy(selection_mask).float()
                return selection_mask

            labels = torch.LongTensor([int(entry['label']) for
                                       entry in input_list])
            segment_ids_list = [entry['segment_ids'] for entry in input_list]
            segment_ids, _ = pad_sentences(
                segment_ids_list, context_length=self.context_length)
            sent_list = [entry['sent_ids'] for entry in input_list]
            sent_tensor, pad_attn_mask = pad_sentences(
                sent_list, context_length=self.context_length)

            # print(pad_attn_mask, pad_attn_mask.size())
            seq_len = pad_attn_mask.size(1)
            b_size = pad_attn_mask.size(0)
            vocab_size = self.tokenizer.vocab_size
            # print(b_size, seq_len)

            # Transform the pad attention mask to 3D tensor of shape:
            # (b_size, seq_len, seq_len)
            pad_attn_mask = pad_attn_mask.unsqueeze(dim=1).repeat(
                1, seq_len, 1)

            candidate_pos_ids_list = [entry['candidate_pos_ids']
                                      for entry in input_list]
            candidate2_pos_ids_list = [entry['candidate2_pos_ids']
                                       for entry in input_list]
            mask_pos_ids = [entry['mask_id'] for entry in input_list]

            # Get selection masks of both candidates
            cand_select_mask1 = get_candidate_selection_mask(
                selected_pos_ids_list=candidate_pos_ids_list)
            cand_select_mask2 = get_candidate_selection_mask(
                selected_pos_ids_list=candidate2_pos_ids_list)
            # Get selection mask for the "mask" token emb
            cand_select_mask0 = get_candidate_selection_mask(
                selected_pos_ids_list=mask_pos_ids)

            candidate_ids_list = [entry['candidate_ids']
                                  for entry in input_list]
            candidate2_ids_list = [entry['candidate2_ids']
                                   for entry in input_list]
            mask_ids = [entry['mask_id'] for entry in input_list]
            # print(candidate_ids_list)
            #print(self.tokenizer.convert_ids_to_tokens(
            #    candidate_ids_list[0]))
            #print(self.tokenizer.convert_ids_to_tokens(
            #    candidate2_ids_list[0]))

            lik_select_mask1 = get_likelihood_selection_mask(
                selected_ids_list=candidate_ids_list,
                mask_id_list=mask_ids)
            lik_select_mask2 = get_likelihood_selection_mask(
                selected_ids_list=candidate2_ids_list,
                mask_id_list=mask_ids)

            attention_mask = pad_attn_mask

            return_dict = {
                'sentence': sent_tensor,
                'lik_select_mask1': lik_select_mask1,
                'lik_select_mask2': lik_select_mask2,
                'cand_select_mask1': cand_select_mask1,
                'cand_select_mask2': cand_select_mask2,
                'cand_select_mask0': cand_select_mask0,
                'attention_mask': attention_mask,
                'token_type_ids': segment_ids
            }

            return return_dict, labels

        if self.do_train:
            self.data_loader = DataLoader(
                self.dataset, batch_size=self.batch_size,
                shuffle=True, collate_fn=collate_fn,
                num_workers=8, pin_memory=True,
                worker_init_fn=np.random.seed(2809),
                drop_last=self.drop_last)
        else:
            self.data_loader = DataLoader(self.dataset,
                                          batch_size=self.batch_size,
                                          collate_fn=collate_fn,
                                          num_workers=8, pin_memory=True,
                                          worker_init_fn=np.random.seed(2809),
                                          drop_last=drop_last)


# This fn converts a list of sentences (as list of word ids),
# to padded sentences tensor, along with BERT attention mask
def pad_sentences(sent_list, context_length, pad_idx=0):
    b_size = len(sent_list)
    sent_lens = [len(sent) for sent in sent_list]
    max_sent_len = max(sent_lens)
    assert max_sent_len <= context_length

    def pad_x(x_):
        # Pad the data entry (list) to max_input_length (pad_id=-1)
        x_ = x_ + [pad_idx for _ in range(max_sent_len - len(x_))]
        x_ = torch.LongTensor(x_).unsqueeze(0)
        return x_

    padded_sentences = torch.cat([pad_x(sent) for sent in sent_list])

    """
    BertModel (huggingface repo) documentation: attention_mask:
    an optional torch.LongTensor of shape [batch_size, sequence_length]
    with indices selected in [0, 1]. It's a mask to be used if
    some input sequence lengths are smaller than the max input
    sequence length of the current batch. It's the mask that we
    typically use for attention when a batch has varying length
    sentences.
    """
    mask = torch.zeros([b_size, max_sent_len], dtype=torch.long)
    for i in range(b_size):
        for j in range(max_sent_len):
            if j < sent_lens[i]:
                mask[i][j] += 1
    return padded_sentences, mask


def log_weights(model, step, tb_writer):
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.view(-1).mean().item()
            std = param.data.view(-1).std().item()
            tb_writer.add_scalar('Param {}/Weight mean'.format(name), 
                                 mean, step + 1)
            tb_writer.add_scalar('Param {}/Weight std'.format(name),
                                 std, step + 1)
