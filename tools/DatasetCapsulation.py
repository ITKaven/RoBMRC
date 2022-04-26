from torch.utils.data import Dataset, DataLoader
import numpy as np


class QueryAndAnswer:
    def __init__(self, line, forward_asp_query, forward_opi_query,
                 forward_asp_query_mask, forward_asp_query_seg,
                 forward_opi_query_mask, forward_opi_query_seg,
                 forward_asp_answer_start, forward_asp_answer_end,
                 forward_opi_answer_start, forward_opi_answer_end,
                 backward_asp_query, backward_opi_query,
                 backward_asp_answer_start, backward_asp_answer_end,
                 backward_asp_query_mask, backward_asp_query_seg,
                 backward_opi_query_mask, backward_opi_query_seg,
                 backward_opi_answer_start, backward_opi_answer_end,
                 sentiment_query, sentiment_answer,
                 sentiment_query_mask, sentiment_query_seg):
        self.line = line
        self.forward_asp_query = forward_asp_query
        self.forward_opi_query = forward_opi_query
        self.forward_asp_query_mask = forward_asp_query_mask
        self.forward_asp_query_seg = forward_asp_query_seg
        self.forward_opi_query_mask = forward_opi_query_mask
        self.forward_opi_query_seg = forward_opi_query_seg
        self.forward_asp_answer_start = forward_asp_answer_start
        self.forward_asp_answer_end = forward_asp_answer_end
        self.forward_opi_answer_start = forward_opi_answer_start
        self.forward_opi_answer_end = forward_opi_answer_end
        self.backward_asp_query = backward_asp_query
        self.backward_opi_query = backward_opi_query
        self.backward_asp_query_mask = backward_asp_query_mask
        self.backward_asp_query_seg = backward_asp_query_seg
        self.backward_opi_query_mask = backward_opi_query_mask
        self.backward_opi_query_seg = backward_opi_query_seg
        self.backward_asp_answer_start = backward_asp_answer_start
        self.backward_asp_answer_end = backward_asp_answer_end
        self.backward_opi_answer_start = backward_opi_answer_start
        self.backward_opi_answer_end = backward_opi_answer_end
        self.sentiment_query = sentiment_query
        self.sentiment_answer = sentiment_answer
        self.sentiment_query_mask = sentiment_query_mask
        self.sentiment_query_seg = sentiment_query_seg


class TestDataset:
    def __init__(self, line, aspect_list, opinion_list, asp_opi_list, asp_sent_list, triplet_list):
        self.line = line
        self.aspect_list = aspect_list
        self.opinion_list = opinion_list
        self.asp_opi_list = asp_opi_list
        self.asp_sent_list = asp_sent_list
        self.triplet_list = triplet_list


class ReviewDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        dataset_to_numpy_array = {
            'line': self.dataset[item].line,
            'forward_asp_query': np.array(self.dataset[item].forward_asp_query),
            'forward_opi_query': np.array(self.dataset[item].forward_opi_query),
            'forward_asp_query_mask': np.array(self.dataset[item].forward_asp_query_mask),
            'forward_opi_query_mask': np.array(self.dataset[item].forward_opi_query_mask),
            'forward_asp_query_seg': np.array(self.dataset[item].forward_asp_query_seg),
            'forward_opi_query_seg': np.array(self.dataset[item].forward_opi_query_seg),
            'forward_asp_answer_start': np.array(self.dataset[item].forward_asp_answer_start),
            'forward_asp_answer_end': np.array(self.dataset[item].forward_asp_answer_end),
            'forward_opi_answer_start': np.array(self.dataset[item].forward_opi_answer_start),
            'forward_opi_answer_end': np.array(self.dataset[item].forward_opi_answer_end),
            'backward_asp_query': np.array(self.dataset[item].backward_asp_query),
            'backward_opi_query': np.array(self.dataset[item].backward_opi_query),
            'backward_asp_query_mask': np.array(self.dataset[item].backward_asp_query_mask),
            'backward_opi_query_mask': np.array(self.dataset[item].backward_opi_query_mask),
            'backward_asp_query_seg': np.array(self.dataset[item].backward_asp_query_seg),
            'backward_opi_query_seg': np.array(self.dataset[item].backward_opi_query_seg),
            'backward_asp_answer_start': np.array(self.dataset[item].backward_asp_answer_start),
            'backward_asp_answer_end': np.array(self.dataset[item].backward_asp_answer_end),
            'backward_opi_answer_start': np.array(self.dataset[item].backward_opi_answer_start),
            'backward_opi_answer_end': np.array(self.dataset[item].backward_opi_answer_end),
            'sentiment_query': np.array(self.dataset[item].sentiment_query),
            'sentiment_answer': np.array(self.dataset[item].sentiment_answer),
            'sentiment_query_mask': np.array(self.dataset[item].sentiment_query_mask),
            'sentiment_query_seg': np.array(self.dataset[item].sentiment_query_seg)
        }
        return dataset_to_numpy_array

    def get_batch_num(self, batch_size):
        if len(self.dataset) % batch_size == 0:
            return len(self.dataset) / batch_size
        return int(len(self.dataset) / batch_size) + 1


def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, gpu=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)
    for data_dict in dataloader:
        _dict = {}
        for name, tensor in data_dict.items():
            if gpu and name != 'line':
                _dict[name] = data_dict[name].cuda()
            else:
                _dict[name] = data_dict[name]
        yield _dict
