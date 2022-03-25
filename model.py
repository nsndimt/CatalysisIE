import logging
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AdamW, get_linear_schedule_with_warmup
from pytorch_lightning.utilities.seed import seed_everything
from seqeval.metrics import classification_report, f1_score
from utils import *


class BERT(nn.Module):

    def __init__(self, source_path, dropout):
        super(BERT, self).__init__()
        self.dimension = 768
        self._repr_model = AutoModel.from_pretrained(source_path,
                                                     hidden_dropout_prob=dropout,
                                                     attention_probs_dropout_prob=dropout
                                                     )

    def forward(self, pad_subword_tokens, pad_attn_mask, pad_word_start, word_size):
        outputs = self._repr_model(pad_subword_tokens, attention_mask=pad_attn_mask)
        all_hidden = outputs[0]

        batch_size, _, hidden_dim = all_hidden.size()
        _, start_num = pad_word_start.size()
        positions = pad_word_start.unsqueeze(-1).expand(batch_size, start_num, hidden_dim)
        return torch.gather(all_hidden, dim=-2, index=positions)


class Biaffine(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(Biaffine, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.W1 = nn.Linear(input_dim, hidden_dim)
        self.W2 = nn.Linear(input_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)

        self.W_bilin_weight = nn.Parameter(torch.Tensor(hidden_dim + 1, hidden_dim + 1, output_dim))
        self.W_bilin_weight.data.zero_()

    def forward(self, con_repr, gather_start=None, gather_end=None):
        if gather_start is not None:
            batch_size, token_num, hidden_dim = con_repr.size()
            start = con_repr.view(batch_size*token_num, hidden_dim)
            start = torch.index_select(start, 0, gather_start)
            end = con_repr.view(batch_size*token_num, hidden_dim)
            end = torch.index_select(end, 0, gather_end)

            start_rep = self.dropout(F.relu(self.W1(start)))
            end_rep = self.dropout(F.relu(self.W2(end)))

            start_rep = torch.cat([start_rep, start_rep.new_ones(*start_rep.size()[:-1], 1)], len(start_rep.size()) - 1)
            end_rep = torch.cat([end_rep, end_rep.new_ones(*end_rep.size()[:-1], 1)], len(end_rep.size()) - 1)

            # (S x D1) * (D1 x (D2 x O)) -> S x (D2 x O)
            intermediate = start_rep.mm(self.W_bilin_weight.view(-1, (self.hidden_dim + 1) * self.output_dim))
            # (S x 1 x D2) * (S x D2 x O) -> S x (1 x O)
            output = end_rep.unsqueeze(1).bmm(intermediate.view(-1, self.hidden_dim + 1, self.output_dim)).view(-1, self.output_dim)

            return torch.log_softmax(output,dim=-1)
        else:
            batch_size, token_num, hidden_dim = con_repr.size()
            start_rep = self.dropout(F.relu(self.W1(con_repr)))
            end_rep = self.dropout(F.relu(self.W2(con_repr)))

            start_rep = torch.cat([start_rep, start_rep.new_ones(*start_rep.size()[:-1], 1)], len(start_rep.size()) - 1)
            end_rep = torch.cat([end_rep, end_rep.new_ones(*end_rep.size()[:-1], 1)], len(end_rep.size()) - 1)

            # ((N x L1) x D1) * (D1 x (D2 x O)) -> (N x L1) x (D2 x O)
            intermediate = torch.mm(start_rep.view(-1, self.hidden_dim + 1),
                                    self.W_bilin_weight.view(-1, (self.hidden_dim + 1) * self.output_dim))
            # (N x L2 x D2) -> (N x D2 x L2)
            end_rep_T = end_rep.transpose(1, 2)
            # ((N x L1) x (D2 x O)) -> (N x L1 x O x D2)
            intermediate_T = intermediate.view(batch_size, token_num, self.hidden_dim + 1,  self.output_dim).transpose(2, 3)
            # (N x (L1 x O) x D2) * (N x D2 x L2) -> (N x (L1 x O) x L2)
            output = intermediate_T.reshape(batch_size, token_num * self.output_dim, (self.hidden_dim + 1)).bmm(end_rep_T)
            # (N x (L1 x O) x L2) -> (N x L1 x L2 x O)
            output = output.view(batch_size, token_num, self.output_dim, token_num).transpose(2, 3)

            return torch.log_softmax(output,dim=-1)


class BERTSpan(pl.LightningModule):

    def __init__(self, model_name, train_dataset, val_dataset, test_dataset, labels, batch_size, neg_rate,
                 hidden_dim, dropout_rate, lr, wd, epoch_num):
        super(BERTSpan, self).__init__()
        self.save_hyperparameters('dropout_rate', 'hidden_dim', 'lr', 'wd', 'epoch_num', 'neg_rate',
                                  'batch_size', 'labels')
        self.lr = lr
        self.wd = wd
        self.epoch_num = epoch_num
        self.neg_rate = neg_rate
        self.batch_size = batch_size

        self.model_name = model_name
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.labels = labels
        self.output_label_encoder = init_crf_label_encoder(self.labels,'BIO')
        self.label_encoder = init_span_label_encoder(self.labels)

        self.val_report = []
        self.test_report = []

        seed_everything(12345)
        self._encoder = BERT(model_name, dropout_rate)
        seed_everything(12345)
        self._span_criterion = Biaffine(self._encoder.dimension, hidden_dim, len(self.label_encoder), dropout_rate)

    def decode(self, class_log, word_size):
        listing_it = np.argmax(class_log, axis=-1)
        listing_vt = np.max(class_log, axis=-1)
        outputs = []

        for l_mat, v_mat, sent_l in zip(listing_it, listing_vt, word_size):
            sample_candidates = []
            #O should be zero
            xidx, yidx = np.nonzero(l_mat)
            for i, j in zip(xidx, yidx):
                if i <= j < sent_l:
                    label = self.label_encoder.get(int(l_mat[i][j]))
                    value = v_mat[i][j]
                    sample_candidates.append((i, j, label, value))

            ordered_seg = sorted(sample_candidates, key=lambda x: -x[-1])
            filter_list = []
            for elem in ordered_seg:
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append(elem)
            outputs.append(filter_list)

        return outputs

    def forward(self, pad_subword_tokens, pad_attn_mask, pad_word_start, word_size):
        con_repr = self._encoder(pad_subword_tokens, pad_attn_mask, pad_word_start, word_size)
        class_logit = self._span_criterion(con_repr)
        return  class_logit.detach().cpu().numpy()

    def embed(self, pad_subword_tokens, pad_attn_mask, pad_word_start, word_size):
        con_repr = self._encoder(pad_subword_tokens, pad_attn_mask, pad_word_start, word_size)
        norm_span_rep = self._span_extractor(con_repr)
        cos_sim = self._span_criterion.predict(norm_span_rep)
        return norm_span_rep.detach().cpu().numpy(), cos_sim.detach().cpu().numpy()

    def training_step(self, batch, batch_idx):
        raw_data, data = batch
        pad_subword_tokens, pad_attn_mask, pad_word_start, gather_start, gather_end, label_list, word_size = data

        con_repr = self._encoder(pad_subword_tokens, pad_attn_mask, pad_word_start, word_size)
        class_logit = self._span_criterion(con_repr, gather_start, gather_end)
        loss = F.nll_loss(class_logit, label_list)
        self.log('train_loss', loss)
        return loss

    def eval_step(self, batch):
        assert self.training == False
        batch_idx, data = batch
        pad_subword_tokens, pad_attn_mask, pad_word_start, word_size = data

        outputs = self.decode(self(pad_subword_tokens, pad_attn_mask, pad_word_start, word_size), word_size)

        outputs_encoded = []
        for filter_list, sent_l in zip(outputs, word_size):
            pred_label = iob_tagging(filter_list, sent_l)
            outputs_encoded.append([self.output_label_encoder.index(tag) for tag in pred_label])

        max_size = max(word_size)
        pad_sizes = [max_size - len(o) for o in outputs_encoded]
        outputs_encoded = torch.LongTensor([o + [0] * p for o, p in zip(outputs_encoded, pad_sizes)])

        batch_idx = torch.LongTensor(batch_idx).unsqueeze(-1)
        word_size = torch.LongTensor(word_size).unsqueeze(-1)
        info_tensor = torch.cat([batch_idx, word_size, outputs_encoded], dim=-1)
        return info_tensor.detach()

    def eval_epoch(self, step_outputs, dataset):
        outputs, oracles = [], []
        for batch_output in step_outputs:
            batch_output = batch_output.cpu().numpy().tolist()
            batch_idxs = [output[0] for output in batch_output]
            sent_sizes = [output[1] for output in batch_output]
            pred_labels = [output[2:2 + sl] for output, sl in zip(batch_output, sent_sizes)]
            for idx, sl, pred_label in zip(batch_idxs, sent_sizes, pred_labels):
                sent_i, subsent =dataset.get_raw(idx)
                pred_label = [self.output_label_encoder.get(idx) for idx in pred_label]
                sent_tag = [token['label'] for token in subsent]
                dataset.set_pred(idx, pred_label)

                outputs.append(pred_label)
                oracles.append(sent_tag)
        return outputs, oracles

    def validation_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def validation_epoch_end(self, validation_step_outputs):
        outputs, oracles = self.eval_epoch(validation_step_outputs, self.span_val_dataset)

        logging.info('\n'+classification_report(oracles, outputs, digits=4)+'\n')
        logging.info('\n'+soft_classification_report(oracles, outputs)+'\n')
        self.val_report.append((
            classification_report(oracles, outputs, output_dict=True),
            soft_classification_report(oracles, outputs, output_dict=True)
        ))
        micro_f1 = self.val_report[-1][0]['micro avg']['f1-score']
        self.log('hp_metric', micro_f1)

    def test_step(self, batch, batch_idx):
        return self.eval_step(batch)

    def test_epoch_end(self, test_step_outputs):
        outputs, oracles = self.eval_epoch(test_step_outputs, self.span_test_dataset)
        
        logging.info('#### Strict Match Report ####')
        logging.info('\n'+classification_report(oracles, outputs, digits=4)+'\n')
        logging.info('#### Soft Match Report ####')
        logging.info('\n'+soft_classification_report(oracles, outputs)+'\n')
        self.test_report.append((
            classification_report(oracles, outputs, output_dict=True),
            soft_classification_report(oracles, outputs, output_dict=True)
        ))
        micro_f1 = self.test_report[-1][0]['micro avg']['f1-score']
        self.log('micro_f1', micro_f1)

    def pred_dataset_step(self, offset, batch, dataset):
        assert self.training == False
        batch_idx, data = batch
        pad_subword_tokens, pad_attn_mask, pad_word_start, word_size = data
        class_logit = self(pad_subword_tokens, pad_attn_mask, pad_word_start, word_size)
        decoded=self.decode(class_logit, word_size)

        for i, (idx, sl, filter_list) in enumerate(zip(batch_idx, word_size, decoded)):
            assert idx == (offset+i), (offset, i)
            pred_tag = iob_tagging(filter_list, sl)
            dataset.set_pred(idx, pred_tag)

    def batch_cuda(self, batch):
        batch_idx, data = batch
        pad_subword_tokens, pad_attn_mask, pad_word_start, word_size = data
        return batch_idx, (pad_subword_tokens.cuda(), pad_attn_mask.cuda(), pad_word_start.cuda(), word_size)

    def setup(self, stage):
        self.span_train_dataset = SpanTrainDataset(self.train_dataset, self.model_name, self.label_encoder,
                                                   self.neg_rate)
        self.span_val_dataset = SpanTestDataset(self.val_dataset, self.model_name, self.label_encoder)
        self.span_test_dataset = SpanTestDataset(self.test_dataset, self.model_name, self.label_encoder)
        if len(self.span_train_dataset) % self.batch_size == 0:
            self.train_loader_len = len(self.span_train_dataset) // self.batch_size
        else:
            self.train_loader_len = len(self.span_train_dataset) // self.batch_size + 1
        self.total_steps = int(self.train_loader_len * self.epoch_num)
        warmup_proportion = 0.1
        self.warmup_steps = int(self.total_steps * warmup_proportion)

    def train_dataloader(self):
        seed_everything(12345)
        loader = DataLoader(self.span_train_dataset, self.batch_size, shuffle=True,
                            collate_fn=span_train_collect_fn, num_workers=8)
        assert len(loader) == self.train_loader_len
        return loader

    def val_dataloader(self):
        seed_everything(12345)
        return DataLoader(self.span_val_dataset, self.batch_size, collate_fn=span_test_collect_fn, num_workers=8)

    def test_dataloader(self):
        seed_everything(12345)
        return DataLoader(self.span_test_dataset, self.batch_size, collate_fn=span_test_collect_fn, num_workers=8)

    def gen_pred_dataloader(self, sents):
        seed_everything(12345)
        pred_dataset = SpanTestDataset(sents, self.model_name, self.label_encoder)
        pred_dataloader = DataLoader(pred_dataset, self.batch_size, collate_fn=span_test_collect_fn)
        return pred_dataset, pred_dataloader

    def configure_optimizers(self):
        all_parameters = list(self.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_param = [
            {'params': [p for n, p in all_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': self.wd},
            {'params': [p for n, p in all_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        optimizer = AdamW(grouped_param, lr=self.lr)

        logging.info(f'epoch_num:{self.epoch_num} total step: {self.total_steps} warmup step: {self.warmup_steps}')
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps)
        logging.info(f'lr:{self.lr} wd:{self.wd} batch:{self.batch_size}')

        return [
            {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        ]