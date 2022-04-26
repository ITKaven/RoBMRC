from transformers import BertModel
import torch.nn as nn


class BERTModel(nn.Module):
    def __init__(self, hidden_size, bert_model_type):

        super(BERTModel, self).__init__()

        # BERT模型
        if bert_model_type == 'bert-base-uncased':
            self.bert = BertModel.from_pretrained(bert_model_type)
            print('bert-base-uncased model loaded')

        else:
            raise KeyError('bert_model_type should be bert-based-uncased.')

        self.classifier_a_start = nn.Linear(hidden_size, 2)
        self.classifier_a_end = nn.Linear(hidden_size, 2)
        self.classifier_ao_start = nn.Linear(hidden_size, 2)
        self.classifier_ao_end = nn.Linear(hidden_size, 2)
        self.classifier_o_start = nn.Linear(hidden_size, 2)
        self.classifier_o_end = nn.Linear(hidden_size, 2)
        self.classifier_oa_start = nn.Linear(hidden_size, 2)
        self.classifier_oa_end = nn.Linear(hidden_size, 2)
        self.classifier_sentiment = nn.Linear(hidden_size, 3)

    def forward(self, query_tensor, query_mask, query_seg, step):

        hidden_states = self.bert(query_tensor, attention_mask=query_mask, token_type_ids=query_seg)[0]
        if step == 'A':
            predict_start = self.classifier_a_start(hidden_states)
            predict_end = self.classifier_a_end(hidden_states)
            return predict_start, predict_end
        elif step == 'O':
            predict_start = self.classifier_o_start(hidden_states)
            predict_end = self.classifier_o_end(hidden_states)
            return predict_start, predict_end
        elif step == 'AO':
            predict_start = self.classifier_ao_start(hidden_states)
            predict_end = self.classifier_ao_end(hidden_states)
            return predict_start, predict_end
        elif step == 'OA':
            predict_start = self.classifier_oa_start(hidden_states)
            predict_end = self.classifier_oa_end(hidden_states)
            return predict_start, predict_end
        elif step == 'S':
            sentiment_hidden_states = hidden_states[:, 0, :]
            sentiment_scores = self.classifier_sentiment(sentiment_hidden_states)
            return sentiment_scores
        else:
            raise KeyError('step error.')
