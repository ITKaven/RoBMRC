import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
from torch.nn import functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
import DatasetCapsulation as Data
from tools import Utils, Model
from transformers import BertTokenizer

dataset_version = "v1/"
dataset_name_list = ["14res", "14lap", "15res", "16res"]
dataset_type_list = ["train_triplets", "dev_triplets", "test_triplets"]
if dataset_version.__eq__("v1/"):
    dataset_name_list = ["14rest", "14lap", "15rest", "16rest"]
    dataset_type_list = ["train", "dev", "test"]
inference_beta = [0.90, 0.90, 0.90, 0.90]


def test(model, tokenize, batch_generator, test_data, beta, logger, gpu, max_len):
    model.eval()

    triplet_target_num = 0
    asp_target_num = 0
    opi_target_num = 0
    asp_opi_target_num = 0
    asp_sent_target_num = 0

    triplet_predict_num = 0
    asp_predict_num = 0
    opi_predict_num = 0
    asp_opi_predict_num = 0
    asp_sent_predict_num = 0

    triplet_match_num = 0
    asp_match_num = 0
    opi_match_num = 0
    asp_opi_match_num = 0
    asp_sent_match_num = 0

    for batch_index, batch_dict in enumerate(batch_generator):

        triplets_target = test_data[batch_index].triplet_list
        asp_target = test_data[batch_index].aspect_list
        opi_target = test_data[batch_index].opinion_list
        asp_opi_target = test_data[batch_index].asp_opi_list
        asp_sent_target = test_data[batch_index].asp_sent_list

        # 预测三元组
        triplets_predict = []
        asp_predict = []
        opi_predict = []
        asp_opi_predict = []
        asp_sent_predict = []

        forward_pair_list = []
        forward_pair_prob = []
        forward_pair_ind_list = []

        backward_pair_list = []
        backward_pair_prob = []
        backward_pair_ind_list = []

        final_asp_list = []
        final_opi_list = []
        final_asp_ind_list = []
        final_opi_ind_list = []

        ok_start_index = batch_dict['forward_asp_answer_start'][0].gt(-1).float().nonzero()

        ok_start_tokens = batch_dict['forward_asp_query'][0][ok_start_index].squeeze(1)

        f_asp_start_scores, f_asp_end_scores = model(batch_dict['forward_asp_query'],
                                                     batch_dict['forward_asp_query_mask'],
                                                     batch_dict['forward_asp_query_seg'], 'A')

        f_asp_start_scores = F.softmax(f_asp_start_scores[0], dim=1)
        f_asp_end_scores = F.softmax(f_asp_end_scores[0], dim=1)

        f_asp_start_prob, f_asp_start_ind = torch.max(f_asp_start_scores, dim=1)
        f_asp_end_prob, f_asp_end_ind = torch.max(f_asp_end_scores, dim=1)

        f_asp_start_prob_temp = []
        f_asp_end_prob_temp = []
        f_asp_start_index_temp = []
        f_asp_end_index_temp = []
        for start_index in range(f_asp_start_ind.size(0)):
            if batch_dict['forward_asp_answer_start'][0, start_index] != -1:
                if f_asp_start_ind[start_index].item() == 1:
                    f_asp_start_index_temp.append(start_index)
                    f_asp_start_prob_temp.append(f_asp_start_prob[start_index].item())
                if f_asp_end_ind[start_index].item() == 1:
                    f_asp_end_index_temp.append(start_index)
                    f_asp_end_prob_temp.append(f_asp_end_prob[start_index].item())

        f_asp_start_index, f_asp_end_index, f_asp_prob = Utils.filter_unpaired(
            f_asp_start_prob_temp, f_asp_end_prob_temp, f_asp_start_index_temp, f_asp_end_index_temp, max_len)

        # 根据预测到的切面，生成查询
        for start_index in range(len(f_asp_start_index)):
            opinion_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What opinion given the aspect'.split(' ')])
            for j in range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1):
                opinion_query.append(batch_dict['forward_asp_query'][0][j].item())
            opinion_query.append(tokenize.convert_tokens_to_ids('?'))
            opinion_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            opinion_query_seg = [0] * len(opinion_query)
            f_opi_length = len(opinion_query)

            opinion_query = torch.tensor(opinion_query).long()
            if gpu:
                opinion_query = opinion_query.cuda()
            opinion_query = torch.cat([opinion_query, ok_start_tokens], -1).unsqueeze(0)
            opinion_query_seg += [1] * ok_start_tokens.size(0)
            opinion_query_mask = torch.ones(opinion_query.size(1)).float().unsqueeze(0)
            if gpu:
                opinion_query_mask = opinion_query_mask.cuda()
            opinion_query_seg = torch.tensor(opinion_query_seg).long().unsqueeze(0)
            if gpu:
                opinion_query_seg = opinion_query_seg.cuda()

            f_opi_start_scores, f_opi_end_scores = model(opinion_query, opinion_query_mask, opinion_query_seg, 'AO')

            f_opi_start_scores = F.softmax(f_opi_start_scores[0], dim=1)
            f_opi_end_scores = F.softmax(f_opi_end_scores[0], dim=1)
            f_opi_start_prob, f_opi_start_ind = torch.max(f_opi_start_scores, dim=1)
            f_opi_end_prob, f_opi_end_ind = torch.max(f_opi_end_scores, dim=1)

            f_opi_start_prob_temp = []
            f_opi_end_prob_temp = []
            f_opi_start_index_temp = []
            f_opi_end_index_temp = []
            for k in range(f_opi_start_ind.size(0)):
                if opinion_query_seg[0, k] == 1:
                    if f_opi_start_ind[k].item() == 1:
                        f_opi_start_index_temp.append(k)
                        f_opi_start_prob_temp.append(f_opi_start_prob[k].item())
                    if f_opi_end_ind[k].item() == 1:
                        f_opi_end_index_temp.append(k)
                        f_opi_end_prob_temp.append(f_opi_end_prob[k].item())

            f_opi_start_index, f_opi_end_index, f_opi_prob = Utils.filter_unpaired(
                f_opi_start_prob_temp, f_opi_end_prob_temp, f_opi_start_index_temp, f_opi_end_index_temp, max_len)

            for idx in range(len(f_opi_start_index)):
                asp = [batch_dict['forward_asp_query'][0][j].item() for j in
                       range(f_asp_start_index[start_index], f_asp_end_index[start_index] + 1)]
                opi = [opinion_query[0][j].item() for j in range(f_opi_start_index[idx], f_opi_end_index[idx] + 1)]
                asp_ind = [f_asp_start_index[start_index] - 5, f_asp_end_index[start_index] - 5]
                opi_ind = [f_opi_start_index[idx] - f_opi_length, f_opi_end_index[idx] - f_opi_length]
                # TODO
                temp_prob = math.sqrt(f_asp_prob[start_index] * f_opi_prob[idx])
                if asp_ind + opi_ind not in forward_pair_list:
                    forward_pair_list.append([asp] + [opi])
                    forward_pair_prob.append(temp_prob)
                    forward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)

        b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                     batch_dict['backward_opi_query_mask'],
                                                     batch_dict['backward_opi_query_seg'], 'O')
        b_opi_start_scores = F.softmax(b_opi_start_scores[0], dim=1)
        b_opi_end_scores = F.softmax(b_opi_end_scores[0], dim=1)
        b_opi_start_prob, b_opi_start_ind = torch.max(b_opi_start_scores, dim=1)
        b_opi_end_prob, b_opi_end_ind = torch.max(b_opi_end_scores, dim=1)

        b_opi_start_prob_temp = []
        b_opi_end_prob_temp = []
        b_opi_start_index_temp = []
        b_opi_end_index_temp = []
        for start_index in range(b_opi_start_ind.size(0)):
            if batch_dict['backward_opi_answer_start'][0, start_index] != -1:
                if b_opi_start_ind[start_index].item() == 1:
                    b_opi_start_index_temp.append(start_index)
                    b_opi_start_prob_temp.append(b_opi_start_prob[start_index].item())
                if b_opi_end_ind[start_index].item() == 1:
                    b_opi_end_index_temp.append(start_index)
                    b_opi_end_prob_temp.append(b_opi_end_prob[start_index].item())

        b_opi_start_index, b_opi_end_index, b_opi_prob = Utils.filter_unpaired(
            b_opi_start_prob_temp, b_opi_end_prob_temp, b_opi_start_index_temp, b_opi_end_index_temp, max_len)

        for start_index in range(len(b_opi_start_index)):
            aspect_query = tokenize.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                 '[CLS] What aspect does the opinion'.split(' ')])
            for j in range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1):
                aspect_query.append(batch_dict['backward_opi_query'][0][j].item())
            aspect_query.append(tokenize.convert_tokens_to_ids('describe'))
            aspect_query.append(tokenize.convert_tokens_to_ids('?'))
            aspect_query.append(tokenize.convert_tokens_to_ids('[SEP]'))
            aspect_query_seg = [0] * len(aspect_query)
            b_asp_length = len(aspect_query)
            aspect_query = torch.tensor(aspect_query).long()
            if gpu:
                aspect_query = aspect_query.cuda()
            aspect_query = torch.cat([aspect_query, ok_start_tokens], -1).unsqueeze(0)
            aspect_query_seg += [1] * ok_start_tokens.size(0)
            aspect_query_mask = torch.ones(aspect_query.size(1)).float().unsqueeze(0)
            if gpu:
                aspect_query_mask = aspect_query_mask.cuda()
            aspect_query_seg = torch.tensor(aspect_query_seg).long().unsqueeze(0)
            if gpu:
                aspect_query_seg = aspect_query_seg.cuda()

            b_asp_start_scores, b_asp_end_scores = model(aspect_query, aspect_query_mask, aspect_query_seg, 'OA')

            b_asp_start_scores = F.softmax(b_asp_start_scores[0], dim=1)
            b_asp_end_scores = F.softmax(b_asp_end_scores[0], dim=1)
            b_asp_start_prob, b_asp_start_ind = torch.max(b_asp_start_scores, dim=1)
            b_asp_end_prob, b_asp_end_ind = torch.max(b_asp_end_scores, dim=1)

            b_asp_start_prob_temp = []
            b_asp_end_prob_temp = []
            b_asp_start_index_temp = []
            b_asp_end_index_temp = []
            for k in range(b_asp_start_ind.size(0)):
                if aspect_query_seg[0, k] == 1:
                    if b_asp_start_ind[k].item() == 1:
                        b_asp_start_index_temp.append(k)
                        b_asp_start_prob_temp.append(b_asp_start_prob[k].item())
                    if b_asp_end_ind[k].item() == 1:
                        b_asp_end_index_temp.append(k)
                        b_asp_end_prob_temp.append(b_asp_end_prob[k].item())

            b_asp_start_index, b_asp_end_index, b_asp_prob = Utils.filter_unpaired(
                b_asp_start_prob_temp, b_asp_end_prob_temp, b_asp_start_index_temp, b_asp_end_index_temp, max_len)

            for idx in range(len(b_asp_start_index)):
                opi = [batch_dict['backward_opi_query'][0][j].item() for j in
                       range(b_opi_start_index[start_index], b_opi_end_index[start_index] + 1)]
                asp = [aspect_query[0][j].item() for j in range(b_asp_start_index[idx], b_asp_end_index[idx] + 1)]
                asp_ind = [b_asp_start_index[idx] - b_asp_length, b_asp_end_index[idx] - b_asp_length]
                opi_ind = [b_opi_start_index[start_index] - 5, b_opi_end_index[start_index] - 5]
                # TODO
                temp_prob = math.sqrt(b_asp_prob[idx] * b_opi_prob[start_index])
                if asp_ind + opi_ind not in backward_pair_ind_list:
                    backward_pair_list.append([asp] + [opi])
                    backward_pair_prob.append(temp_prob)
                    backward_pair_ind_list.append(asp_ind + opi_ind)
                else:
                    print('error')
                    exit(1)

        for idx in range(len(forward_pair_list)):
            if forward_pair_list[idx] in backward_pair_list or forward_pair_prob[idx] >= beta:
                if forward_pair_list[idx][0] not in final_asp_list:
                    final_asp_list.append(forward_pair_list[idx][0])
                    final_opi_list.append([forward_pair_list[idx][1]])
                    final_asp_ind_list.append(forward_pair_ind_list[idx][:2])
                    final_opi_ind_list.append([forward_pair_ind_list[idx][2:]])
                else:
                    asp_index = final_asp_list.index(forward_pair_list[idx][0])
                    if forward_pair_list[idx][1] not in final_opi_list[asp_index]:
                        final_opi_list[asp_index].append(forward_pair_list[idx][1])
                        final_opi_ind_list[asp_index].append(forward_pair_ind_list[idx][2:])
        # backward
        for idx in range(len(backward_pair_list)):
            if backward_pair_list[idx] not in forward_pair_list:
                if backward_pair_prob[idx] >= beta:
                    if backward_pair_list[idx][0] not in final_asp_list:
                        final_asp_list.append(backward_pair_list[idx][0])
                        final_opi_list.append([backward_pair_list[idx][1]])
                        final_asp_ind_list.append(backward_pair_ind_list[idx][:2])
                        final_opi_ind_list.append([backward_pair_ind_list[idx][2:]])
                    else:
                        asp_index = final_asp_list.index(backward_pair_list[idx][0])
                        if backward_pair_list[idx][1] not in final_opi_list[asp_index]:
                            final_opi_list[asp_index].append(backward_pair_list[idx][1])
                            final_opi_ind_list[asp_index].append(backward_pair_ind_list[idx][2:])

        # sentiment
        for idx in range(len(final_asp_list)):
            predict_opinion_num = len(final_opi_list[idx])
            for idy in range(predict_opinion_num):
                sentiment_query = tokenize.convert_tokens_to_ids(
                    [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in
                     '[CLS] What sentiment given the aspect'.split(' ')])
                sentiment_query += final_asp_list[idx]
                sentiment_query += tokenize.convert_tokens_to_ids(
                    [word.lower() for word in 'and the opinion'.split(' ')])
                sentiment_query += final_opi_list[idx][idy]
                sentiment_query.append(tokenize.convert_tokens_to_ids('?'))
                sentiment_query.append(tokenize.convert_tokens_to_ids('[SEP]'))

                sentiment_query_seg = [0] * len(sentiment_query)
                sentiment_query = torch.tensor(sentiment_query).long()
                if gpu:
                    sentiment_query = sentiment_query.cuda()
                sentiment_query = torch.cat([sentiment_query, ok_start_tokens], -1).unsqueeze(0)
                sentiment_query_seg += [1] * ok_start_tokens.size(0)
                sentiment_query_mask = torch.ones(sentiment_query.size(1)).float().unsqueeze(0)
                if gpu:
                    sentiment_query_mask = sentiment_query_mask.cuda()
                sentiment_query_seg = torch.tensor(sentiment_query_seg).long().unsqueeze(0)
                if gpu:
                    sentiment_query_seg = sentiment_query_seg.cuda()

                sentiment_scores = model(sentiment_query, sentiment_query_mask, sentiment_query_seg, 'S')
                sentiment_predicted = torch.argmax(sentiment_scores[0], dim=0).item()

                asp_f = []
                opi_f = []
                asp_f.append(final_asp_ind_list[idx][0])
                asp_f.append(final_asp_ind_list[idx][1])
                opi_f.append(final_opi_ind_list[idx][idy][0])
                opi_f.append(final_opi_ind_list[idx][idy][1])
                triplet_predict = asp_f + opi_f + [sentiment_predicted]
                if opi_f not in opi_predict:
                    opi_predict.append(opi_f)
                if asp_f + opi_f not in asp_opi_predict:
                    asp_opi_predict.append(asp_f + opi_f)
                if asp_f + [sentiment_predicted] not in asp_sent_predict:
                    asp_sent_predict.append(asp_f + [sentiment_predicted])
                if asp_f not in asp_predict:
                    asp_predict.append(asp_f)
                if triplet_predict not in triplets_predict:
                    triplets_predict.append(triplet_predict)

        triplet_target_num += len(triplets_target)
        asp_target_num += len(asp_target)
        opi_target_num += len(opi_target)
        asp_opi_target_num += len(asp_opi_target)
        asp_sent_target_num += len(asp_sent_target)

        triplet_predict_num += len(triplets_predict)
        asp_predict_num += len(asp_predict)
        opi_predict_num += len(opi_predict)
        asp_opi_predict_num += len(asp_opi_predict)
        asp_sent_predict_num += len(asp_sent_predict)

        for trip in triplets_predict:
            for trip_ in triplets_target:
                if trip_ == trip:
                    triplet_match_num += 1
        for trip in asp_predict:
            for trip_ in asp_target:
                if trip_ == trip:
                    asp_match_num += 1
        for trip in opi_predict:
            for trip_ in opi_target:
                if trip_ == trip:
                    opi_match_num += 1
        for trip in asp_opi_predict:
            for trip_ in asp_opi_target:
                if trip_ == trip:
                    asp_opi_match_num += 1
        for trip in asp_sent_predict:
            for trip_ in asp_sent_target:
                if trip_ == trip:
                    asp_sent_match_num += 1

    precision = float(triplet_match_num) / float(triplet_predict_num + 1e-6)
    recall = float(triplet_match_num) / float(triplet_target_num + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    logger.info('Triplet - Precision: {}\tRecall: {}\tF1: {}'.format(precision, recall, f1))

    precision_aspect = float(asp_match_num) / float(asp_predict_num + 1e-6)
    recall_aspect = float(asp_match_num) / float(asp_target_num + 1e-6)
    f1_aspect = 2 * precision_aspect * recall_aspect / (precision_aspect + recall_aspect + 1e-6)
    logger.info('Aspect - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect, recall_aspect, f1_aspect))

    precision_opinion = float(opi_match_num) / float(opi_predict_num + 1e-6)
    recall_opinion = float(opi_match_num) / float(opi_target_num + 1e-6)
    f1_opinion = 2 * precision_opinion * recall_opinion / (precision_opinion + recall_opinion + 1e-6)
    logger.info('Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_opinion, recall_opinion, f1_opinion))

    precision_aspect_sentiment = float(asp_sent_match_num) / float(asp_sent_predict_num + 1e-6)
    recall_aspect_sentiment = float(asp_sent_match_num) / float(asp_sent_target_num + 1e-6)
    f1_aspect_sentiment = 2 * precision_aspect_sentiment * recall_aspect_sentiment / (
            precision_aspect_sentiment + recall_aspect_sentiment + 1e-6)
    logger.info('Aspect-Sentiment - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_sentiment,
                                                                              recall_aspect_sentiment,
                                                                              f1_aspect_sentiment))

    precision_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_predict_num + 1e-6)
    recall_aspect_opinion = float(asp_opi_match_num) / float(asp_opi_target_num + 1e-6)
    f1_aspect_opinion = 2 * precision_aspect_opinion * recall_aspect_opinion / (
            precision_aspect_opinion + recall_aspect_opinion + 1e-6)
    logger.info(
        'Aspect-Opinion - Precision: {}\tRecall: {}\tF1: {}'.format(precision_aspect_opinion, recall_aspect_opinion,
                                                                    f1_aspect_opinion))
    return f1


def create_directory(arguments):
    if not os.path.exists(arguments.log_path):
        os.mkdir(arguments.log_path)
        os.mkdir(arguments.log_path + dataset_version)
    if not os.path.exists(arguments.save_model_path):
        os.mkdir(arguments.save_model_path)
        os.mkdir(arguments.save_model_path + dataset_version)
    log_path = arguments.log_path + dataset_version + arguments.data_name + arguments.model_name + '.log'
    model_path = arguments.save_model_path + dataset_version + arguments.data_name + arguments.model_name + '.pth'
    if not os.path.exists(log_path):
        log = open(log_path, 'w')
        log.close()
        model = open(model_path, 'w')
        model.close()


def train(arguments):
    create_directory(arguments)

    log_path = arguments.log_path + dataset_version + arguments.data_name + arguments.model_name + '.log'
    model_path = arguments.save_model_path + dataset_version + arguments.data_name + arguments.model_name + '.pth'

    # init logger and tokenize
    logger, fh, sh = Utils.get_logger(log_path)
    tokenize = BertTokenizer.from_pretrained(arguments.bert_model_type)

    # load data
    logger.info('loading data......')
    train_data_path = arguments.data_path + arguments.data_name + '.pt'
    test_data_path = arguments.data_path + arguments.data_name + '_test.pt'

    train_total_data = torch.load(train_data_path)
    test_total_data = torch.load(test_data_path)

    train_data = train_total_data[arguments.train]
    dev_data = train_total_data[arguments.dev]
    test_data = train_total_data[arguments.test]
    max_len = train_total_data[arguments.max_len]
    max_aspect_num = train_total_data[arguments.max_aspect_num]

    train_standard = test_total_data[arguments.train]
    dev_standard = test_total_data[arguments.dev]
    test_standard = test_total_data[arguments.test]

    model = Model.BERTModel(arguments.hidden_size, arguments.bert_model_type)
    if arguments.gpu:
        model = model.cuda()

    if arguments.mode == 'test':
        test_dataset = Data.ReviewDataset(test_data)
        # load checkpoint
        logger.info('loading model......')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['net'])

        # eval
        logger.info('evaluating......')

        batch_generator_test = Data.generate_batches(dataset=test_dataset, batch_size=1, shuffle=False,
                                                     gpu=arguments.gpu)
        test(model, tokenize, batch_generator_test, test_standard, arguments.inference_beta, logger,
             arguments.gpu, max_len)

    elif arguments.mode == 'train':
        train_dataset = Data.ReviewDataset(train_data)
        dev_dataset = Data.ReviewDataset(dev_data)
        test_dataset = Data.ReviewDataset(test_data)
        batch_num_train = train_dataset.get_batch_num(arguments.batch_size)

        # optimizer
        logger.info('initial optimizer......')
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if "bert" in n], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if "bert" not in n],
             'lr': arguments.learning_rate, 'weight_decay': 0.01}]
        optimizer = AdamW(optimizer_grouped_parameters, lr=arguments.tuning_bert_rate, correct_bias=False)

        # load saved model, optimizer and epoch num
        if arguments.reload and os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info('Reload model and optimizer after training epoch {}'.format(checkpoint['epoch']))
        else:
            start_epoch = 1
            logger.info('New model and optimizer from epoch 1')

        # scheduler
        training_steps = arguments.epoch_num * batch_num_train
        warmup_steps = int(training_steps * arguments.warm_up)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=training_steps)

        # training
        logger.info('begin training......')
        best_f1 = 0.
        for epoch in range(start_epoch, arguments.epoch_num + 1):
            model.train()
            model.zero_grad()

            batch_generator = Data.generate_batches(dataset=train_dataset, batch_size=arguments.batch_size,
                                                    gpu=arguments.gpu)

            for batch_index, batch_dict in enumerate(batch_generator):

                optimizer.zero_grad()

                f_aspect_start_scores, f_aspect_end_scores = model(batch_dict['forward_asp_query'],
                                                                   batch_dict['forward_asp_query_mask'],
                                                                   batch_dict['forward_asp_query_seg'], 'A')

                f_asp_loss = Utils.calculate_entity_loss(f_aspect_start_scores, f_aspect_end_scores,
                                                         batch_dict['forward_asp_answer_start'],
                                                         batch_dict['forward_asp_answer_end'], arguments.gpu)

                b_opi_start_scores, b_opi_end_scores = model(batch_dict['backward_opi_query'],
                                                             batch_dict['backward_opi_query_mask'],
                                                             batch_dict['backward_opi_query_seg'], 'O')

                b_opi_loss = Utils.calculate_entity_loss(b_opi_start_scores, b_opi_end_scores,
                                                         batch_dict['backward_opi_answer_start'],
                                                         batch_dict['backward_opi_answer_end'], arguments.gpu)

                batch_dict['forward_opi_query'] = batch_dict['forward_opi_query'] \
                    .view(-1, batch_dict['forward_opi_query'].size(-1))
                batch_dict['forward_opi_query_mask'] = batch_dict['forward_opi_query_mask'] \
                    .view(-1, batch_dict['forward_opi_query_mask'].size(-1))
                batch_dict['forward_opi_query_seg'] = batch_dict['forward_opi_query_seg'] \
                    .view(-1, batch_dict['forward_opi_query_seg'].size(-1))

                f_opi_start_scores, f_opi_end_scores = model(
                    batch_dict['forward_opi_query'],
                    batch_dict['forward_opi_query_mask'],
                    batch_dict['forward_opi_query_seg'],
                    'AO')

                f_opi_loss = Utils.calculate_entity_loss(f_opi_start_scores, f_opi_end_scores,
                                                         batch_dict['forward_opi_answer_start'].view(-1, batch_dict[
                                                             'forward_opi_answer_start'].size(-1)),
                                                         batch_dict['forward_opi_answer_end'].view(-1, batch_dict[
                                                             'forward_opi_answer_end'].size(-1)), arguments.gpu
                                                         ) / max_aspect_num

                # TODO b_asp
                batch_dict['backward_asp_query'] = batch_dict['backward_asp_query'] \
                    .view(-1, batch_dict['backward_asp_query'].size(-1))
                batch_dict['backward_asp_query_mask'] = batch_dict['backward_asp_query_mask'] \
                    .view(-1, batch_dict['backward_asp_query_mask'].size(-1))
                batch_dict['backward_asp_query_seg'] = batch_dict['backward_asp_query_seg'] \
                    .view(-1, batch_dict['backward_asp_query_seg'].size(-1))

                b_asp_start_scores, b_asp_end_scores = model(
                    batch_dict['backward_asp_query'],
                    batch_dict['backward_asp_query_mask'],
                    batch_dict['backward_asp_query_seg'],
                    'OA')

                b_asp_loss = Utils.calculate_entity_loss(
                    b_asp_start_scores, b_asp_end_scores,
                    batch_dict['backward_asp_answer_start'].view(-1, batch_dict[
                        'backward_asp_answer_start'].size(-1)),
                    batch_dict['backward_asp_answer_end'].view(-1, batch_dict[
                        'backward_asp_answer_end'].size(-1)), arguments.gpu
                ) / max_aspect_num

                # TODO sentiment
                batch_dict['sentiment_query'] = batch_dict['sentiment_query'] \
                    .view(-1, batch_dict['sentiment_query'].size(-1))
                batch_dict['sentiment_query_mask'] = batch_dict['sentiment_query_mask'] \
                    .view(-1, batch_dict['sentiment_query_mask'].size(-1))
                batch_dict['sentiment_query_seg'] = batch_dict['sentiment_query_seg'] \
                    .view(-1, batch_dict['sentiment_query_seg'].size(-1))

                sentiment_scores = model(
                    batch_dict['sentiment_query'],
                    batch_dict['sentiment_query_mask'],
                    batch_dict['sentiment_query_seg'],
                    'S')

                sentiment_loss = Utils.calculate_sentiment_loss(
                    sentiment_scores,
                    batch_dict['sentiment_answer'].view(-1)) / max_aspect_num

                # loss
                loss_sum = f_asp_loss + f_opi_loss + b_opi_loss + b_asp_loss + \
                           arguments.beta * sentiment_loss
                loss_sum.backward()
                optimizer.step()
                scheduler.step()

                # train logger
                if (batch_index + 1) % 10 == 0:
                    logger.info('Epoch:[{}/{}]\t Batch:[{}/{}]\t Loss Sum:{}\t '
                                'forward Loss:{};{}\t backward Loss:{};{}\t Sentiment Loss:{}\t'.
                                format(epoch, arguments.epoch_num, batch_index + 1, batch_num_train,
                                       round(loss_sum.item(), 4),
                                       round(f_asp_loss.item(), 4), round(f_opi_loss.item(), 4),
                                       round(b_opi_loss.item(), 4), round(b_asp_loss.item(), 4),
                                       round(sentiment_loss.item(), 4)))

            # validation
            batch_generator_dev = Data.generate_batches(dataset=dev_dataset, batch_size=1, shuffle=False,
                                                        gpu=arguments.gpu)
            logger.info("dev")
            test(model, tokenize, batch_generator_dev, dev_standard, arguments.inference_beta, logger,
                 arguments.gpu, max_len)

            # test
            batch_generator_test = Data.generate_batches(dataset=test_dataset, batch_size=1, shuffle=False,
                                                         gpu=arguments.gpu)
            logger.info("test")
            f1_test = test(model, tokenize, batch_generator_test, test_standard, arguments.inference_beta, logger,
                           arguments.gpu, max_len)
            # save model and optimizer
            if f1_test > best_f1:
                best_f1 = f1_test
                logger.info('Model saved after epoch {}'.format(epoch))
                state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, model_path)
    else:
        logger.info('Error mode!')
        exit(1)
    logger.removeHandler(fh)
    logger.removeHandler(sh)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bidirectional MRC-based sentiment triplet extraction')
    parser.add_argument('--data_path', type=str, default="../data/preprocess/" + dataset_version)
    parser.add_argument('--data_name', type=str, default=dataset_name_list[0], choices=dataset_name_list)
    parser.add_argument('--log_path', type=str, default="../log/")
    parser.add_argument('--save_model_path', type=str, default="../model/")
    parser.add_argument('--model_name', type=str, default="_BiMRC")

    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--train', type=str, default=dataset_type_list[0], choices=dataset_type_list)
    parser.add_argument('--dev', type=str, default=dataset_type_list[1], choices=dataset_type_list)
    parser.add_argument('--test', type=str, default=dataset_type_list[2], choices=dataset_type_list)
    parser.add_argument('--max_len', type=str, default="max_len", choices=["max_len"])
    parser.add_argument('--max_aspect_num', type=str, default="max_aspect_num", choices=["max_aspect_num"])

    parser.add_argument('--reload', type=bool, default=False)

    parser.add_argument('--bert_model_type', type=str, default="bert-base-uncased")
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--inference_beta', type=float, default=0.90)

    # training hyper-parameter
    parser.add_argument('--gpu', type=bool, default=True)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--tuning_bert_rate', type=float, default=1e-5)
    parser.add_argument('--warm_up', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=1)

    args = parser.parse_args()
    for i in range(len(dataset_name_list)):
        args.data_name = dataset_name_list[i]
        args.inference_beta = inference_beta[i]
        train(args)
