import random
import re
import logging
from collections import defaultdict
from copy import deepcopy

import pytorch_lightning as pl
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer


class LabelEncoder(object):
    def __init__(self):
        self.idx_to_item = []
        self.item_to_idx = {}

    def add(self, item):
        if item not in self.item_to_idx:
            self.item_to_idx[item] = len(self.idx_to_item)
            self.idx_to_item.append(item)

    def get(self, idx):
        return self.idx_to_item[idx]

    def index(self, item):
        return self.item_to_idx[item]

    def __len__(self):
        return len(self.item_to_idx)

    def __str__(self):
        return str(self.item_to_idx)


def init_span_label_encoder(labels, add_empty=True):
    label_encoder = LabelEncoder()
    if add_empty:
        labels = ['O'] + labels
    for label in labels:
        label_encoder.add(label)
    return label_encoder


def init_crf_label_encoder(labels, tags):
    label_encoder = LabelEncoder()
    tag_labels = ['O'] + [f'{tag}-{label}' for tag in tags for label in labels if tag != 'O']
    for tag_label in tag_labels:
        label_encoder.add(tag_label)
    return label_encoder


def get_bio_spans(labels):
    prev_tag = 'O'
    prev_type = None
    begin_offset = 0
    spans = []

    token_info = []

    for label in labels:
        if label == 'O':
            token_info.append(('O', None))
        else:
            token_info.append((label[0], label[2:]))
    token_info.append(('O', None))

    for i, (token_tag, token_type) in enumerate(token_info):
        if (prev_tag, token_tag) in [('B', 'B'), ('B', 'O'), ('I', 'B'), ('I', 'O')]:
            spans.append((begin_offset, i - 1, prev_type))
        if token_tag == 'B':
            begin_offset = i

        prev_tag = token_tag
        prev_type = token_type

    return spans


def soft_classification_report(oracles, outputs, output_dict=False):
    def span_value(i, j, l):
        return 1

    def span_match(i, j, l, a, b, c):
        if l == c:
            if j >= a and i <= b:
                left = max(i, a)
                right = min(j, b)
                return (right - left + 1) / (j - i + 1)
            else:
                return 0
        else:
            return 0

    def single_side(spans_a, spans_b):
        deposit = {}
        gain = 0
        for i, j, l in spans_a:
            deposit[(i, j, l)] = span_value(i, j, l)
            for a, b, c in spans_b:
                match_score = span_match(i, j, l, a, b, c)
                gain += match_score * deposit[(i, j, l)]
        total_gain = sum(deposit.values())
        return gain, total_gain

    all_output_spans, all_oracle_spans = [], []
    oracle_labels = set()
    for sent_output, sent_oracle in zip(outputs, oracles):
        span_output = get_bio_spans(sent_output)
        span_oracle = get_bio_spans(sent_oracle)
        all_output_spans.append(span_output)
        all_oracle_spans.append(span_oracle)
        oracle_labels.update([l for i, j, l in span_oracle])

    res_dict = {}
    for target_l in oracle_labels:
        p_numerator, p_denominator = 0, 0
        r_numerator, r_denominator = 0, 0
        for span_output, span_oracle in zip(all_output_spans, all_oracle_spans):
            filter_span_output = [(i, j, l) for i, j, l in span_output if l == target_l]
            filter_span_oracle = [(i, j, l) for i, j, l in span_oracle if l == target_l]
            p_gain = single_side(filter_span_output, filter_span_oracle)
            p_numerator += p_gain[0]
            p_denominator += p_gain[1]
            r_gain = single_side(filter_span_oracle, filter_span_output)
            r_numerator += r_gain[0]
            r_denominator += r_gain[1]

        p = p_numerator / p_denominator if p_denominator != 0 else 0.
        r = r_numerator / r_denominator if r_denominator != 0 else 0.
        f1 = (2 * p * r) / (p + r) if (p + r) != 0 else 0.
        res_dict[target_l] = {'precision': p, 'recall': r, 'f1-score': f1, 'support': r_denominator,
                              'pred': p_denominator}

    p_numerator, p_denominator = 0, 0
    r_numerator, r_denominator = 0, 0
    for span_output, span_oracle in zip(all_output_spans, all_oracle_spans):
        p_gain = single_side(span_output, span_oracle)
        p_numerator += p_gain[0]
        p_denominator += p_gain[1]
        r_gain = single_side(span_oracle, span_output)
        r_numerator += r_gain[0]
        r_denominator += r_gain[1]

    p = p_numerator / p_denominator if p_denominator != 0 else 0.
    r = r_numerator / r_denominator if r_denominator != 0 else 0.
    f1 = (2 * p * r) / (p + r) if (p + r) != 0 else 0.
    res_dict['micro avg'] = {'precision': p, 'recall': r, 'f1-score': f1, 'support': r_denominator,
                             'pred': p_denominator}

    res_dict['macro avg'] = {
        k: np.mean([res_dict[label][k] for label in oracle_labels]).item() for k in ['precision', 'recall', 'f1-score']
    }
    res_dict['macro avg']['support'] = np.sum([res_dict[label]['support'] for label in oracle_labels]).item()
    res_dict['macro avg']['pred'] = np.sum([res_dict[label]['pred'] for label in oracle_labels]).item()

    if output_dict:
        return res_dict
    else:
        report = f"{'':>20}{'precision':>9}{'recall':>9}{'f1-score':>9}{'support':>9}\n"
        for k in sorted(list(oracle_labels))+['micro avg','macro avg']:
            line = res_dict[k]
            precison, recall, f1, support, pred = line['precision'], line['recall'], line['f1-score'], line['support'], line['pred']
            report += f'{k:<20}{precison:9.4f}{recall:9.4f}{f1:9.4f}{support:>9}\n'
        return report


def tokenize_split(tokenizer, sent, max_size=382):
    sent = deepcopy(sent)
    start = 0
    subword_start = 0

    token_buf = []
    res = []

    non_empty = []
    for i, token in enumerate(sent):
        text = token['text']
        tag = token['label']
        subword_token = tokenizer.tokenize(text)
        token['subword'] = subword_token
        if len(subword_token) == 0:
            if tag == 'O':
                print(f'empty subword {text} {tag}')
        else:
            non_empty.append(token)


    for token in non_empty:
        subword_token = token['subword']
        if subword_start + len(subword_token) <= max_size:
            token_buf.append(token)
            subword_start += len(subword_token)
        else:
            res.append(token_buf)
            subword_start = len(subword_token)
            token_buf = [token]

    if len(token_buf) > 0:
        res.append(token_buf)

    return res


class SpanTrainDataset(Dataset):
    def __init__(self, sents, model_name, label_encoder, neg_rate):
        super(SpanTrainDataset, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.neg_rate = neg_rate
        self.label_encoder = label_encoder

        self.raw = []
        self.processed = []

        for i, sent in enumerate(sents):
            for subsent in tokenize_split(self.tokenizer, sent):
                self.raw.append((i, subsent))
                self.processed.append(self.preprocess(subsent))

    def preprocess(self, sent):
        cls_sign = '[CLS]'  # hard code
        sep_sign = '[SEP]'

        subword_tokens, word_start = [], []
        for token in sent:
            subword_token = token['subword']
            word_start.append(len(subword_tokens))
            subword_tokens.extend(subword_token)
        subword_tokens = [cls_sign] + subword_tokens + [sep_sign]
        word_start = [1 + pos for pos in word_start]
        assert all([not subword_tokens[idx].startswith('##') for idx in word_start]), 'wrong word start'
        subword_tokens = self.tokenizer.convert_tokens_to_ids(subword_tokens)
        assert len(subword_tokens) <= 512, len(subword_tokens)
        assert len(word_start) == len(sent)

        spans = get_bio_spans([token['label'] for token in sent])
        pos_span= []
        for start, end, label in spans:
            pos_span.append((start, end, self.label_encoder.index(label)))

        reject_set = [(start, end) for start, end, label in spans]
        word_size = len(sent)
        neg_span = [(i, j, self.label_encoder.index("O")) for i in range(word_size) for j in range(i, word_size) if (i, j) not in reject_set]

        return subword_tokens, word_start, pos_span, neg_span

    def resample(self, processed):
        subword_tokens, word_start, pos_span, neg_span = processed

        word_size = len(word_start)
        if len(neg_span) > 0 and self.neg_rate > 0:
            neg_num = int(word_size * self.neg_rate) + 1
            sample_num = min(neg_num, len(neg_span))
            sampled_neg_span = random.sample(neg_span, sample_num)
        elif self.neg_rate == 0:
            sampled_neg_span = []
        else:
            sampled_neg_span = neg_span
        return subword_tokens, word_start, pos_span, sampled_neg_span

    def __getitem__(self, item):
        sampled = self.resample(self.processed[item])
        return item, sampled

    def __len__(self):
        return len(self.raw)

    def get_raw(self, idx):
        return self.raw[idx]


def span_train_collect_fn(batch):
    pad_sign = 0  # hard code <PAD> id is 0
    batch_idx = [item[0] for item in batch]
    sents = [item[1] for item in batch]

    sizes = [len(sent[0]) for sent in sents]
    max_size = max(sizes)
    pad_sizes = [max_size - size for size in sizes]
    pad_subword_tokens = [sent[0] + [pad_sign] * pad_size for sent, pad_size in zip(sents, pad_sizes)]

    word_sizes = [len(sent[1]) for sent in sents]
    max_word_size = max(word_sizes)
    pad_word_sizes = [max_word_size - size for size in word_sizes]
    pad_word_start = [sent[1] + [max_size - 1] * pad_size for sent, pad_size in zip(sents, pad_word_sizes)]

    pad_subword_tokens = torch.LongTensor(pad_subword_tokens)
    pad_attn_mask = torch.LongTensor([[1] * size + [0] * pad_size for size, pad_size in zip(sizes, pad_sizes)])
    assert pad_subword_tokens.size() == pad_attn_mask.size()
    pad_word_start = torch.LongTensor(pad_word_start)

    gather_start = []
    gather_end = []
    label_list = []
    for i, sent in enumerate(sents):
        for j, (m, n, label) in enumerate(sent[2] + sent[3]):
            gather_start.append(i * max_word_size + m)
            gather_end.append(i * max_word_size + n)
            label_list.append(label)

    gather_start = torch.LongTensor(gather_start)
    gather_end = torch.LongTensor(gather_end)
    label_list = torch.LongTensor(label_list)

    assert gather_start.size() == gather_end.size() and gather_start.size() == label_list.size()

    return batch_idx, (pad_subword_tokens, pad_attn_mask, pad_word_start, gather_start, gather_end, label_list, word_sizes)


class SpanTestDataset(Dataset):
    def __init__(self, sents, source_path, label_encoder):
        super(SpanTestDataset, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(source_path, use_fast=False)
        self.label_encoder = label_encoder
        self.label_name = [self.label_encoder.get(i) for i in range(len(self.label_encoder))]

        self.raw = []
        self.processed = []
        self.pred = []
        for i, sent in enumerate(sents):
            for subsent in tokenize_split(self.tokenizer, sent):
                self.raw.append((i, subsent))
                self.processed.append(self.preprocess(subsent))
                self.pred.append(None)

    def preprocess(self, sent):
        cls_sign = '[CLS]'  # hard code
        sep_sign = '[SEP]'

        subword_tokens, word_start = [], []
        for token in sent:
            subword_token = token['subword']
            word_start.append(len(subword_tokens))
            subword_tokens.extend(subword_token)
        subword_tokens = [cls_sign] + subword_tokens + [sep_sign]
        word_start = [1 + pos for pos in word_start]
        assert all([not subword_tokens[idx].startswith('##') for idx in word_start]), 'wrong word start'
        subword_tokens = self.tokenizer.convert_tokens_to_ids(subword_tokens)
        assert len(subword_tokens) <= 512, len(subword_tokens)

        return subword_tokens, word_start

    def __getitem__(self, item):
        return item, self.processed[item]

    def __len__(self):
        return len(self.raw)

    def get_raw(self, idx):
        return self.raw[idx]

    def set_pred(self, idx, pred_label):
        self.pred[idx] = pred_label

    def output_pred(self):
        output_dataset = defaultdict(list)
        assert len(self.raw) == len(self.pred)
        for (i, subsent), pred_label in zip(self.raw, self.pred):
            subsent = deepcopy(subsent)
            for token, token_pred in zip(subsent, pred_label):
                token['pred'] = token_pred
            output_dataset[i].extend(subsent)
        output_sents = [output_dataset[i] for i in range(len(output_dataset))]
        return output_sents


def span_test_collect_fn(batch):
    pad_sign = 0  # hard code
    batch_idx = [item[0] for item in batch]
    sents = [item[1] for item in batch]

    sizes = [len(sent[0]) for sent in sents]
    max_size = max(sizes)
    pad_sizes = [max_size - size for size in sizes]
    pad_subword_tokens = [sent[0] + [pad_sign] * pad_size for sent, pad_size in zip(sents, pad_sizes)]

    word_sizes = [len(sent[1]) for sent in sents]
    max_word_size = max(word_sizes)
    pad_word_sizes = [max_word_size - size for size in word_sizes]
    pad_word_start = [sent[1] + [max_size - 1] * pad_size for sent, pad_size in zip(sents, pad_word_sizes)]

    pad_subword_tokens = torch.LongTensor(pad_subword_tokens)
    pad_attn_mask = torch.LongTensor([[1] * size + [0] * pad_size for size, pad_size in zip(sizes, pad_sizes)])
    assert pad_subword_tokens.size() == pad_attn_mask.size()
    pad_word_start = torch.LongTensor(pad_word_start)

    return batch_idx, (pad_subword_tokens, pad_attn_mask, pad_word_start, word_sizes)


def conflict_judge(line_x, line_y):
    if line_x[0] == line_y[0]:
        return True
    if line_x[0] < line_y[0]:
        if line_x[1] >= line_y[0]:
            return True
    if line_x[0] > line_y[0]:
        if line_x[0] <= line_y[1]:
            return True
    return False


def iob_tagging(entities, s_len):
    tags = ["O"] * s_len

    for el, er, et, es in entities:
        for i in range(el, er + 1):
            if i == el:
                tags[i] = "B-" + et
            else:
                tags[i] = "I-" + et
    return tags


def recursive_cuda(batch):
    if isinstance(batch, torch.Tensor):
        return batch.cuda()
    elif isinstance(batch, dict):
        return {key: recursive_cuda(batch[key]) for key in batch}
    elif isinstance(batch, list):
        return [recursive_cuda(item) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(recursive_cuda(item) for item in batch)
    else:
        return batch
    

def assemble_token_text(tokens):
    text = ''
    prev_end = None
    for token in tokens:
        if prev_end and token['start'] > prev_end:
            text += f' {token["text"]}'
        else:
            text += token['text']
        prev_end = token['end']
    return text


def cleanup_text(text, ignore=[]):
    """
    Original author: Olga Kononova.
    :param ignore:
    :param text:
    :return: Cleaned text.
    :rtype: str
    """
    symbols_table = {
        "\t": " ",
        "&apos;": "'",
        "<sup>": "^{",
        "</sup>": "}",
        "−": "-",
        "&verbar;": "|",
        "&uparrow;": "↑",
        "&CenterDot;": "·",
        "&percnt;": "%",
        "&angst;": "Å",
        "&sol;": "/",
        "&ndash;": "-",
        "&plus;": "+",
        "&equals;": "=",
        "&PlusMinus;": "±",
        "&approx;": "≈",
        "&ast;": "*",
        "&mid;": "∣",
        "&cir;": "○",
        "&squ;": "□",
        "&circledR;": "®",
        "&LongRightArrow;": "⟶",
        "&hbar;": "ℏ",
        "&Agr;": "Α",
        "&agr;": "α",
        "&Bgr;": "Β",
        "&bgr;": "β",
        "&Dgr;": "Δ",
        "&dgr;": "δ",
        "&EEgr;": "Η",
        "&eegr;": "η",
        "&Egr;": "Ε",
        "&egr;": "ε",
        "&Ggr;": "Γ",
        "&ggr;": "γ",
        "&Igr;": "Ι",
        "&igr;": "ι",
        "&Kgr;": "Κ",
        "&kgr;": "κ",
        "&KHgr;": "Χ",
        "&khgr;": "χ",
        "&Lgr;": "Λ",
        "&lgr;": "λ",
        "&Mgr;": "Μ",
        "&mgr;": "μ",
        "&Ngr;": "Ν",
        "&ngr;": "ν",
        "&Ogr;": "Ο",
        "&ogr;": "ο",
        "&OHgr;": "Ω",
        "&ohgr;": "ω",
        "&Pgr;": "Π",
        "&pgr;": "π",
        "&PHgr;": "Φ",
        "&phgr;": "φ",
        "&PSgr;": "Ψ",
        "&psgr;": "ψ",
        "&Rgr;": "Ρ",
        "&rgr;": "ρ",
        "&Sgr;": "Σ",
        "&sgr;": "σ",
        "&Tgr;": "Τ",
        "&tgr;": "τ",
        "&THgr;": "Θ",
        "&thgr;": "θ",
        "&Ugr;": "Υ",
        "&ugr;": "υ",
        "&Xgr;": "Ξ",
        "&xgr;": "ξ",
        "&Zgr;": "Ζ",
        "&zgr;": "ζ"
    }

    quotes_double = [171, 187, 8220, 8221, 8222, 8223, 8243]
    quotes_single = [8216, 8217, 8218, 8219, 8242, 8249, 8250]
    # hyphens = [8722] + [i for i in range(8208, 8214)]
    hyphens = [173, 8722, ord('\ue5f8'), 727, 12287, 12257] + [i for i in range(8208, 8214)]
    times = [183, 215, 8729]
    spaces = [i for i in range(8192, 8208)] + [160, 8239, 8287, 61472]
    formating = [i for i in range(8288, 8298)] + [i for i in range(8299, 8304)] + [i for i in range(8232, 8239)]
    degrees = [186, 730, 778, 8304, 8728, 9702, 9675]
    # math = [i for i in range(8592, 8961)]
    # modifiers = [i for i in range(688, 768)] + \
    #             [i for i in range(7468, 7531)] + \
    #             [i for i in range(7579, 7616)]
    # combining = [i for i in range(768, 880)] + \
    #             [i for i in range(1156, 1162)] + \
    #             [i for i in range(7616, 7680)] + \
    #             [i for i in range(8400, 8433)]
    # control = [i for i in range(32)] + \
    #           [i for i in range(127, 160)]

    to_remove = [775, 8224, 8234, 8855, 8482, 9839]

    # 8289

    # remove 8298 and next symbol
    # replace 12289 with coma

    new_text = text

    for c in symbols_table:
        new_text = new_text.replace(c, symbols_table[c])

    # hyphens unification
    re_str = ''.join([chr(c) for c in hyphens if c not in ignore])
    re_str = '[' + re_str + ']'
    new_text = re.sub(re_str, chr(45), new_text)

    # spaces unification
    re_str = ''.join([chr(c) for c in spaces if c not in ignore])
    re_str = '[' + re_str + ']'
    new_text = re.sub(re_str, chr(32), new_text)

    # quotes unification
    re_str = ''.join([chr(c) for c in quotes_single if c not in ignore])
    re_str = '[' + re_str + ']'
    new_text = re.sub(re_str, chr(39), new_text)

    re_str = ''.join([chr(c) for c in quotes_double if c not in ignore])
    re_str = '[' + re_str + ']'
    new_text = re.sub(re_str, chr(34), new_text)

    # formatting symbols
    re_str = ''.join([chr(c) for c in formating + to_remove if c not in ignore])
    re_str = '[' + re_str + ']'
    new_text = re.sub(re_str, '', new_text)

    # degrees
    re_str = ''.join([chr(c) for c in degrees if c not in ignore])
    re_str = '[' + re_str + ']'
    new_text = re.sub(re_str, chr(176), new_text)
    new_text = new_text.replace('° C', '°C')
    new_text = new_text.replace('°C', ' °C')

    new_text = re.sub('Fig.\s*([0-9]+)', 'Figure \\1', new_text)
    new_text = re.sub('[Rr]ef.\s*([0-9]+)', 'reference \\1', new_text)
    new_text = re.sub('\sal\.\s*[0-9\s]+', ' al. ', new_text)

    for c in [' Co.', 'Ltd.', 'Inc.', 'A.R.', 'Corp.', 'A. R.']:
        new_text = new_text.replace(c, '')
    new_text = new_text.replace('()', '')

    new_text = re.sub('\s+,', ',', new_text)
    new_text = re.sub('([1-9]),([0-9]{3})', '\\1\\2', new_text)  # remove coma in thousands
    new_text = re.sub('(\w+)=(\d+)', '\\1 = \\2', new_text)
    new_text = re.sub('\(\s+', '(', new_text)
    new_text = new_text.replace(' %', '%')
    new_text = new_text.replace(' )', ')')
    new_text = re.sub('(\d+),([A-Za-z])', '\\1, \\2', new_text)

    new_text = re.sub('\s{2,}', ' ', new_text)

    return new_text


def stanza_fix(sents):
    combined_sents = [sents[0][1]]
    for sent_text, sent in sents[1:]:
        if sent_text[0].isalpha() and sent_text[0].islower() and not (
                sent_text.startswith('i-') or sent_text.startswith('n-')):
            print('combine', ' '.join([t['text'] for t in combined_sents[-1]]), '||',
                  ' '.join([t['text'] for t in sent]))
            combined_sents[-1].extend(sent)
        else:
            combined_sents.append(sent)

    prev_token = None
    merged_sents = []
    for sent in combined_sents:
        merged_sent = []
        for token in sent:
            ifwt = prev_token and prev_token['text'] == 'wt' and token['text'] == '.' and prev_token['end'] == token[
                'start']
            ifna = prev_token and prev_token['text'] == 'Na' and token['text'] == '+' and prev_token['end'] == token[
                'start']
            ifk = prev_token and prev_token['text'] == 'Na' and token['text'] == '+' and prev_token['end'] == token[
                'start']
            if ifwt or ifna or ifk:
                print('merge', prev_token, token)
                merged_sent[-1]['text'] += token['text']
                merged_sent[-1]['end'] = token['end']
                prev_token = merged_sent[-1]
            else:
                merged_sent.append(token)
                prev_token = token
        merged_sents.append(merged_sent)

    idx = 0
    for sent in merged_sents:
        for token in sent:
            token['id'] = idx
            idx += 1

    return merged_sents