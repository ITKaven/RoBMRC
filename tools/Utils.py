import math
import torch
from torch.nn import functional as F
import logging


def normalize_size(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)
    return tensor


def calculate_entity_loss(pred_start, pred_end, gold_start, gold_end, gpu):
    pred_start = normalize_size(pred_start)
    pred_end = normalize_size(pred_end)
    gold_start = normalize_size(gold_start)
    gold_end = normalize_size(gold_end)

    weight = torch.tensor([1, 3]).float()
    if gpu:
        weight = weight.cuda()
    loss_start = F.cross_entropy(pred_start, gold_start.long(), reduction='sum', weight=weight, ignore_index=-1)
    loss_end = F.cross_entropy(pred_end, gold_end.long(), reduction='sum', weight=weight, ignore_index=-1)
    return 0.5 * loss_start + 0.5 * loss_end


def calculate_sentiment_loss(pred_sentiment, gold_sentiment):
    return F.cross_entropy(pred_sentiment, gold_sentiment.long(), reduction='sum', ignore_index=-1)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger, fh, sh


def filter_unpaired(start_prob, end_prob, start, end, max_len):
    filtered_start = []
    filtered_end = []
    filtered_prob = []
    if len(start) > 0 and len(end) > 0:
        length = start[-1] + 1 if start[-1] >= end[-1] else end[-1] + 1
        temp_seq = [0] * length
        for s in start:
            temp_seq[s] += 1
        for e in end:
            temp_seq[e] += 2
        start_index = []
        for idx in range(len(temp_seq)):
            assert temp_seq[idx] < 4
            if temp_seq[idx] == 1:
                start_index.append(idx)
            elif temp_seq[idx] == 2:
                if len(start_index) != 0 and (idx - start_index[-1] + 1) <= max_len:
                    max_prob = 0
                    max_prob_index = 0
                    for index in start_index:
                        if max_prob <= start_prob[start.index(index)] and \
                                (idx - index + 1) <= max_len:
                            max_prob = start_prob[start.index(index)]
                            max_prob_index = index
                    filtered_start.append(max_prob_index)
                    filtered_end.append(idx)
                    filtered_prob.append(math.sqrt(max_prob * end_prob[end.index(idx)]))
                start_index = []
            elif temp_seq[idx] == 3:
                start_index.append(idx)
                max_prob = 0
                max_prob_index = 0
                for index in start_index:
                    if max_prob <= start_prob[start.index(index)] and \
                            (idx - index + 1) <= max_len:
                        max_prob = start_prob[start.index(index)]
                        max_prob_index = index
                filtered_start.append(max_prob_index)
                filtered_end.append(idx)
                filtered_prob.append(math.sqrt(max_prob * end_prob[end.index(idx)]))
                start_index = []
    return filtered_start, filtered_end, filtered_prob
