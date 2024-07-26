import utils
import config
import logging
import numpy as np
from data_process import Processor
from data_loader import MultiTaskDataset, NERPredictDataset
from model import BertMultiTask
from train import train, evaluate, predict_label

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW

import warnings

import pandas as pd
import json

warnings.filterwarnings('ignore')


def dev_split(train_dir, dev_dir):
    """split dev set"""
    train_data = np.load(train_dir, allow_pickle=True)
    x_train = train_data["words"]
    y_train = train_data["labels"]
    y_train_match = train_data["match"]
    y_train_score = train_data["score"]
    dev_data = np.load(dev_dir, allow_pickle=True)
    x_dev = dev_data["words"]
    y_dev = dev_data["labels"]
    y_dev_match = dev_data["match"]
    y_dev_score = dev_data["score"]
    return x_train, x_dev, y_train, y_dev, y_train_match, y_dev_match, y_train_score, y_dev_score


def test():
    data = np.load(config.test_dir, allow_pickle=True)
    word_test = data["words"]
    label_test = data["labels"]
    label_match_test = data["match"]
    label_score_test = data["score"]
    test_dataset = MultiTaskDataset(word_test, label_test, label_match_test, label_score_test, config)
    logging.info("--------Dataset Build!--------")
    # print("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # print("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertMultiTask.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
        # print("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        # print("--------No model to test !--------")
        return
    val_metrics = evaluate(test_loader, model, mode='test')
    val_f1 = val_metrics['f1']
    logging.info("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    # print("test loss: {}, f1 score: {}".format(val_metrics['loss'], val_f1))
    val_f1_labels = val_metrics['f1_labels']
    val_precision_labels = val_metrics['precision_labels']
    val_recall_labels = val_metrics['recall_labels']
    for label in config.labels:
        logging.info("f1 score of {}: {}".format(label, val_f1_labels[label]))
        logging.info("precision score of {}: {}".format(label, val_precision_labels[label]))
        logging.info("recall score of {}: {}".format(label, val_recall_labels[label]))
        print("f1 score of {}: {}".format(label, val_f1_labels[label]))
        print("precision score of {}: {}".format(label, val_precision_labels[label]))
        print("recall score of {}: {}".format(label, val_recall_labels[label]))
    val_f1_match = val_metrics['f1_match']
    logging.info("match f1 score: {}".format(val_f1_match))
    print("match f1 score: {}".format(val_f1_match))
    val_precision_match = val_metrics['precision_match']
    logging.info("match precision score: {}".format(val_precision_match))
    print("match precision score: {}".format(val_precision_match))
    val_recall_match = val_metrics['recall_match']
    logging.info("match recall score: {}".format(val_recall_match))
    print("match recall score: {}".format(val_recall_match))
    MSE = val_metrics['MSE']
    RMSE = val_metrics['RMSE']
    MAE = val_metrics['MAE']
    r2 = val_metrics['r2']
    logging.info("spatial similarity score MSE: {}".format(MSE))
    logging.info("spatial similarity score RMSE: {}".format(RMSE))
    logging.info("spatial similarity score MAE: {}".format(MAE))
    logging.info("spatial similarity score R2: {}".format(r2))
    print("spatial similarity score MSE: {}".format(MSE))
    print("spatial similarity score RMSE: {}".format(RMSE))
    print("spatial similarity score MAE: {}".format(MAE))
    print("spatial similarity score R2: {}".format(r2))

# 小红书笔记NER预测
def predict(word_test):
    # word_test = np.array([
    #     ['杭', '州', '真', '美', '啊'],
    #     ['我', '爱', '西', '湖', '的', '秋', '天'],
    #     ['北', '京']
    # ])
    test_dataset = NERPredictDataset(word_test, config)
    logging.info("--------Dataset Build!--------")
    # build data_loader
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                             shuffle=False, collate_fn=test_dataset.collate_fn)
    logging.info("--------Get Data-loader!--------")
    # Prepare model
    if config.model_dir is not None:
        model = BertMultiTask.from_pretrained(config.model_dir)
        model.to(config.device)
        logging.info("--------Load model from {}--------".format(config.model_dir))
    else:
        logging.info("--------No model to test !--------")
        return
    output_labels = predict_label(test_loader, model, mode='test')
    return output_labels


def load_dev(mode):
    if mode == 'train':
        # 分离出验证集
        word_train, word_dev, label_train, label_dev, label_match_train, label_match_dev, label_score_train, label_score_dev = dev_split(
            config.train_dir, config.dev_dir)
    elif mode == 'test':
        train_data = np.load(config.train_dir, allow_pickle=True)
        dev_data = np.load(config.test_dir, allow_pickle=True)
        word_train = train_data["words"]
        label_train = train_data["labels"]
        label_match_train = train_data["match"]
        label_score_train = train_data["score"]
        word_dev = dev_data["words"]
        label_dev = dev_data["labels"]
        label_match_dev = dev_data["match"]
        label_score_dev = dev_data["score"]
    else:
        word_train = None
        label_train = None
        label_match_train = None
        label_score_train = None
        word_dev = None
        label_dev = None
        label_match_dev = None
        label_score_dev = None

    return word_train, word_dev, label_train, label_dev, label_match_train, label_match_dev, label_score_train, label_score_dev


def run():
    """train the model"""
    # set the logger
    utils.set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))
    # 处理数据，分离文本和标签
    processor = Processor(config)
    processor.process()
    logging.info("--------Process Done!--------")
    # 分离出验证集
    word_train, word_dev, label_train, label_dev, label_match_train, label_match_dev, label_score_train, label_score_dev = load_dev(
        'train')
    # build dataset
    train_dataset = MultiTaskDataset(word_train, label_train, label_match_train, label_score_train, config)
    dev_dataset = MultiTaskDataset(word_dev, label_dev, label_match_dev, label_score_dev, config)
    logging.info("--------Dataset Build!--------")
    # get dataset size
    train_size = len(train_dataset)
    # build data_loader
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size,
                            shuffle=True, collate_fn=dev_dataset.collate_fn)
    logging.info("--------Get Dataloader!--------")
    # Prepare model
    device = config.device
    model = BertMultiTask.from_pretrained(config.macbert_model, num_labels=len(config.label2id))
    model.to(device)
    # Prepare optimizer
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, bilstm, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        # lstm_optimizer = list(model.bilstm.named_parameters())
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        # classifier_optimizer = list(model.classifier.named_parameters())
        # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        # optimizer_grouped_parameters = [
        #     {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
        #      'weight_decay': config.weight_decay},
        #     {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
        #      'weight_decay': 0.0},
        #     # {'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
        #     #  'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        #     # {'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
        #     #  'lr': config.learning_rate * 5, 'weight_decay': 0.0},
        #     {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
        #      'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
        #     {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
        #      'lr': config.learning_rate * 5, 'weight_decay': 0.0},
        #     # {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        # ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)

    # Train the model
    logging.info("--------Start Training!--------")
    train(train_loader, dev_loader, model, optimizer, scheduler, config.model_dir)


if __name__ == '__main__':
    run()
    # processor = Processor(config)
    # processor.process()
    test()
